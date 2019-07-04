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
			gobot.Every(100*time.Millisecond, func() {
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

		window.IMShow(img)
		if window.WaitKey(10) >= 0 {
			break
		}
	}
}

func trackFace(frame *gocv.Mat) {
	if !tracking {
		return
	}
	W := float64(frame.Cols())
	H := float64(frame.Rows())

	blue := color.RGBA{0, 0, 255, 0}
	red := color.RGBA{255, 0, 0, 0}
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
	}

	if detectSize {
		// Set up the reference distance for the initial face.
		detectSize = false
		refDistance = dist(left, top, right, bottom)
	}
	distance := dist(left, top, right, bottom)
	// This right now checks if the face is leaving the FRAME.
	// But what I want is that it would try to keep the frame in the middle.

	// All I want is to keep it at the same place that it was detected at first.
	// Which means it's up to the user to have to perfect position... And then click, tracking.
	// The drone will keep that position.

	// x axis
	switch {
	case right < W/2:
		drone.CounterClockwise(50)
		whereTo := rect
		whereTo.Max.X = int(right + 50)
		whereTo.Max.Y = int(bottom)
		whereTo.Min.X = int(left)
		whereTo.Min.Y = int(top)
		gocv.Rectangle(frame, whereTo, red, 3)
	case left > W/2:
		drone.Clockwise(50)
		whereTo := rect
		whereTo.Max.X = int(right)
		whereTo.Max.Y = int(bottom)
		whereTo.Min.X = int(left + 50)
		whereTo.Min.Y = int(top)
		gocv.Rectangle(frame, whereTo, red, 3)
	default:
		drone.Clockwise(0)
	}

	// y axis
	switch {
	case top < H/10:
		drone.Up(25)
		whereTo := rect
		whereTo.Max.X = int(right)
		whereTo.Max.Y = int(bottom)
		whereTo.Min.X = int(left)
		whereTo.Min.Y = int(top + 25)
		gocv.Rectangle(frame, whereTo, red, 3)
	case bottom > H-(H/10):
		drone.Down(25)
		whereTo := rect
		whereTo.Max.X = int(right)
		whereTo.Max.Y = int(bottom + 25)
		whereTo.Min.X = int(left)
		whereTo.Min.Y = int(top)
		gocv.Rectangle(frame, whereTo, red, 3)
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
