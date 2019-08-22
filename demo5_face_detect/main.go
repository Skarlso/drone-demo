package main

import (
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"os/exec"
	"strconv"
	"sync/atomic"
	"time"

	"gobot.io/x/gobot"
	"gobot.io/x/gobot/platforms/dji/tello"
	"gobot.io/x/gobot/platforms/joystick"
	"gocv.io/x/gocv"
)

type pair struct {
	x float64
	y float64
}

const (
	frameX    = 400
	frameY    = 300
	frameSize = frameX * frameY * 3
	offset    = 32767.0
)

type original struct {
	loc image.Rectangle
	set bool
}

var (
	// ffmpeg command to decode video stream from drone
	ffmpeg = exec.Command("ffmpeg", "-hwaccel", "auto", "-hwaccel_device", "opencl", "-i", "pipe:0",
		"-pix_fmt", "bgr24", "-s", strconv.Itoa(frameX)+"x"+strconv.Itoa(frameY), "-f", "rawvideo", "pipe:1")
	ffmpegIn, _  = ffmpeg.StdinPipe()
	ffmpegOut, _ = ffmpeg.StdoutPipe()

	// gocv
	window = gocv.NewWindow("Tello")

	// tracking
	tracking                 = false
	detectSize               = false
	distTolerance            = 0.05 * dist(0, 0, frameX, frameY)
	refDistance              float64
	left, top, right, bottom float64

	// drone
	drone      = tello.NewDriver("8890")
	flightData *tello.FlightData

	// joystick
	joyAdaptor                   = joystick.NewAdaptor()
	stick                        = joystick.NewDriver(joyAdaptor, "dualshock4")
	leftX, leftY, rightX, rightY atomic.Value

	// Original face location
	orig = original{set: false}
)

func init() {
	leftX.Store(float64(0.0))
	leftY.Store(float64(0.0))
	rightX.Store(float64(0.0))
	rightY.Store(float64(0.0))

	// process drone events in separate goroutine for concurrency
	go func() {
		handleJoystick()

		if err := ffmpeg.Start(); err != nil {
			fmt.Println(err)
			return
		}

		_ = drone.On(tello.FlightDataEvent, func(data interface{}) {
			// TODO: protect flight data from race condition
			flightData = data.(*tello.FlightData)
		})

		_ = drone.On(tello.ConnectedEvent, func(data interface{}) {
			fmt.Println("Connected")
			_ = drone.StartVideo()
			_ = drone.SetVideoEncoderRate(tello.VideoBitRateAuto)
			_ = drone.SetExposure(0)
			gobot.Every(30*time.Millisecond, func() {
				_ = drone.StartVideo()
			})
		})

		_ = drone.On(tello.VideoFrameEvent, func(data interface{}) {
			pkt := data.([]byte)
			if _, err := ffmpegIn.Write(pkt); err != nil {
				fmt.Println(err)
			}
		})

		robot := gobot.NewRobot("tello",
			[]gobot.Connection{joyAdaptor},
			[]gobot.Device{drone, stick},
		)

		_ = robot.Start()
	}()
}

var classifier gocv.CascadeClassifier

func main() {
	classifier = gocv.NewCascadeClassifier()
	defer classifier.Close()
	xmlFile := "haarcascade_frontalface_default.xml"
	if !classifier.Load(xmlFile) {
		fmt.Printf("Error reading cascade file: %v\n", xmlFile)
		return
	}

	for {
		// get next frame from stream
		buf := make([]byte, frameSize)
		if _, err := io.ReadFull(ffmpegOut, buf); err != nil {
			fmt.Println(err)
			continue
		}
		img, _ := gocv.NewMatFromBytes(frameY, frameX, gocv.MatTypeCV8UC3, buf)
		if img.Empty() {
			continue
		}

		trackFace(&img)
		////
		window.IMShow(img)
		if window.WaitKey(10) >= 0 {
			break
		}
	}
}

func trackFace(frame *gocv.Mat) {
	// Once we detect a face, save that location
	// From there on every face's location will be tried to move closer to the original.
	if !tracking {
		orig.set = false
		return
	}
	blue := color.RGBA{0, 0, 255, 0}
	red := color.RGBA{255, 0, 0, 0}
	if orig.set {
		gocv.Rectangle(frame, orig.loc, red, 3)
	}
	var pt image.Point
	rects := classifier.DetectMultiScale(*frame)
	var rect image.Rectangle
	if len(rects) > 0 {
		biggest := image.Point{}
		for _, re := range rects {
			if re.Size().X > biggest.X && re.Size().Y > biggest.Y {
				biggest = re.Size()
				rect = re
			}
		}
		gocv.Rectangle(frame, rect, blue, 3)

		size := gocv.GetTextSize("Human", gocv.FontHersheyPlain, 1.2, 2)
		pt = image.Pt(rect.Min.X+(rect.Min.X/2)-(size.X/2), rect.Min.Y-2)
		gocv.PutText(frame, "Human", pt, gocv.FontHersheyPlain, 1.2, blue, 2)
		left = float64(rect.Min.X)
		top = float64(rect.Min.Y)
		right = float64(rect.Max.X)
		bottom = float64(rect.Max.Y)
		if !orig.set {
			orig.loc = rect
			orig.set = true
			return
		}
	} else {
		return
	}

	if detectSize {
		// Set up the reference distance for the initial face.
		detectSize = false
		refDistance = dist(left, top, right, bottom)
	}
	distance := dist(left, top, right, bottom)
	// If there is an overlap, we are fine... This is to prevent micro corrections to the flight.
	// Also there because of video latency and error in detemining the new position of the face.
	// As long as the faces overlap, we aren't going to modify the location of the drone.
	//if !rect.Intersect(orig.loc).Eq(image.ZR) {
	//	return
	//}
	// TODO: Check out why this caused it to be slower.

	// x axis
	//}
	//fmt.Println("origin: ", orig.loc)
	//fmt.Println("new: ", rect)
	// Only do this if there are no overlaps between the two rectangles.
	switch {
	case orig.loc.Min.X < rect.Min.X && orig.loc.Max.X < rect.Max.X:
		drone.Clockwise(25)
	case orig.loc.Min.X > rect.Min.X && orig.loc.Max.X > rect.Max.X:
		drone.CounterClockwise(25)
	default:
		drone.Clockwise(0)
	}

	switch {
	case orig.loc.Max.Y < rect.Max.Y && orig.loc.Min.Y < rect.Min.Y:
		drone.Down(25)
	case orig.loc.Min.Y > rect.Min.Y && orig.loc.Max.Y > rect.Max.Y:
		drone.Up(25)
	default:
		drone.Up(0)
	}

	// z axis
	switch {
	case distance < refDistance-distTolerance:
		drone.Forward(20)
	case distance > refDistance+distTolerance:
		drone.Backward(20)
	default:
		drone.Forward(0)
	}
}

func dist(x1, y1, x2, y2 float64) float64 {
	return math.Sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
}

func handleJoystick() {
	stick.On(joystick.CirclePress, func(data interface{}) {
		drone.Forward(0)
		drone.Up(0)
		drone.Clockwise(0)
		tracking = !tracking
		if tracking {
			detectSize = true
			println("tracking")
		} else {
			detectSize = false
			orig.set = false
			println("not tracking")
		}
	})
	stick.On(joystick.SquarePress, func(data interface{}) {
		fmt.Println("battery:", flightData.BatteryPercentage)
	})
	stick.On(joystick.TrianglePress, func(data interface{}) {
		drone.TakeOff()
		println("Takeoff")
	})
	stick.On(joystick.XPress, func(data interface{}) {
		drone.Land()
		println("Land")
	})
	stick.On(joystick.LeftX, func(data interface{}) {
		val := float64(data.(int16))
		leftX.Store(val)
	})

	stick.On(joystick.LeftY, func(data interface{}) {
		val := float64(data.(int16))
		leftY.Store(val)
	})

	stick.On(joystick.RightX, func(data interface{}) {
		val := float64(data.(int16))
		rightX.Store(val)
	})

	stick.On(joystick.RightY, func(data interface{}) {
		val := float64(data.(int16))
		rightY.Store(val)
	})
	gobot.Every(50*time.Millisecond, func() {
		rightStick := getRightStick()

		switch {
		case rightStick.x < -10:
			drone.Forward(tello.ValidatePitch(rightStick.x, offset))
		case rightStick.x > 10:
			drone.Backward(tello.ValidatePitch(rightStick.x, offset))
		default:
			drone.Forward(0)
		}

		switch {
		case rightStick.y > 10:
			drone.Right(tello.ValidatePitch(rightStick.y, offset))
		case rightStick.y < -10:
			drone.Left(tello.ValidatePitch(rightStick.y, offset))
		default:
			drone.Right(0)
		}
	})

	gobot.Every(50*time.Millisecond, func() {
		leftStick := getLeftStick()
		switch {
		case leftStick.y < -10:
			drone.Up(tello.ValidatePitch(leftStick.y, offset))
		case leftStick.y > 10:
			drone.Down(tello.ValidatePitch(leftStick.y, offset))
		default:
			drone.Up(0)
		}

		switch {
		case leftStick.x > 20:
			drone.Clockwise(tello.ValidatePitch(leftStick.x, offset))
		case leftStick.x < -20:
			drone.CounterClockwise(tello.ValidatePitch(leftStick.x, offset))
		default:
			drone.Clockwise(0)
		}
	})
}

func getLeftStick() pair {
	s := pair{x: 0, y: 0}
	s.x = leftX.Load().(float64)
	s.y = leftY.Load().(float64)
	return s
}

func getRightStick() pair {
	s := pair{x: 0, y: 0}
	s.x = rightX.Load().(float64)
	s.y = rightY.Load().(float64)
	return s
}
