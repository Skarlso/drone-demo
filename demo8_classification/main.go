// What it does:
//
// This example uses the Caffe (http://caffe.berkeleyvision.org/) deep learning framework
// to classify whatever is in front of the camera.
//
// Download the Caffe model file from:
// http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
//
// Also, you will need the prototxt file:
// https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/bvlc_googlenet.prototxt
//
// And the words text file with the descriptions:
// https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/classification_classes_ILSVRC2012.txt
//
// How to run:
//
// 		go run ./cmd/caffe-classifier/main.go 0 ~/Downloads/bvlc_googlenet.caffemodel ~/Downloads/bvlc_googlenet.prototxt ~/Downloads/classification_classes_ILSVRC2012.txt
//
// You can also use this sample with the Intel OpenVINO Inference Engine, if you have it installed.
//
// 		go run ./cmd/caffe-classifier/main.go 0 ~/Downloads/bvlc_googlenet.caffemodel ~/Downloads/bvlc_googlenet.prototxt ~/Downloads/classification_classes_ILSVRC2012.txt openvino fp16
//
package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
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
	tracking = false

	// drone
	drone      = tello.NewDriver("8890")
	flightData *tello.FlightData

	// joystick
	joyAdaptor                   = joystick.NewAdaptor()
	stick                        = joystick.NewDriver(joyAdaptor, "dualshock4")
	leftX, leftY, rightX, rightY atomic.Value
)

func init() {
	leftX.Store(0.0)
	leftY.Store(0.0)
	rightX.Store(0.0)
	rightY.Store(0.0)

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

func main() {
	// parse args
	model := "bvlc_googlenet.caffemodel"
	config := "bvlc_googlenet.prototxt"
	descr := "classification_classes_ILSVRC2012.txt"
	descriptions, err := readDescriptions(descr)
	if err != nil {
		fmt.Printf("Error reading descriptions file: %v\n", descr)
		return
	}

	backend := gocv.NetBackendDefault
	if len(os.Args) > 5 {
		backend = gocv.ParseNetBackend(os.Args[5])
	}

	target := gocv.NetTargetCPU
	if len(os.Args) > 6 {
		target = gocv.ParseNetTarget(os.Args[6])
	}

	// open DNN classifier
	net := gocv.ReadNet(model, config)
	if net.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, config)
		return
	}
	defer net.Close()
	net.SetPreferableBackend(backend)
	net.SetPreferableTarget(target)

	for {
		buf := make([]byte, frameSize)
		if _, err := io.ReadFull(ffmpegOut, buf); err != nil {
			fmt.Println(err)
			continue
		}
		img, _ := gocv.NewMatFromBytes(frameY, frameX, gocv.MatTypeCV8UC3, buf)
		if img.Empty() {
			continue
		}

		analyse(descriptions, net, &img)

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

func analyse(descriptions []string, net gocv.Net, img *gocv.Mat) {
	status := "Ready"
	statusColor := color.RGBA{0, 255, 0, 0}
	// convert image Mat to 224x224 blob that the classifier can analyze
	blob := gocv.BlobFromImage(*img, 1.0, image.Pt(224, 224), gocv.NewScalar(104, 117, 123, 0), false, false)

	// feed the blob into the classifier
	net.SetInput(blob, "")

	// run a forward pass thru the network
	prob := net.Forward("")

	// reshape the results into a 1x1000 matrix
	probMat := prob.Reshape(1, 1)

	// determine the most probable classification
	_, maxVal, _, maxLoc := gocv.MinMaxLoc(probMat)

	// display classification
	status = fmt.Sprintf("description: %v, maxVal: %v\n", descriptions[maxLoc.X], maxVal)
	gocv.PutText(img, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

	blob.Close()
	prob.Close()
	probMat.Close()
}

// readDescriptions reads the descriptions from a file
// and returns a slice of its lines.
func readDescriptions(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}

func handleJoystick() {
	stick.On(joystick.CirclePress, func(data interface{}) {
		drone.Forward(0)
		drone.Up(0)
		drone.Clockwise(0)
		tracking = !tracking
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
