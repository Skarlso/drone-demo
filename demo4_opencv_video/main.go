package main

import (
	"fmt"
	"image"
	"image/color"
	"io"
	"os/exec"
	"strconv"
	"time"

	"gobot.io/x/gobot"
	"gobot.io/x/gobot/platforms/dji/tello"
	"gocv.io/x/gocv"
)

const (
	frameX    = 960
	frameY    = 720
	frameSize = frameX * frameY * 3
)

func main() {
	drone := tello.NewDriver("8890")
	window := gocv.NewWindow("Tello")

	ffmpeg := exec.Command("ffmpeg", "-hwaccel", "auto", "-hwaccel_device", "opencl", "-i", "pipe:0",
		"-pix_fmt", "bgr24", "-s", strconv.Itoa(frameX)+"x"+strconv.Itoa(frameY), "-f", "rawvideo", "pipe:1")
	ffmpegIn, _ := ffmpeg.StdinPipe()
	ffmpegOut, _ := ffmpeg.StdoutPipe()

	go func() {
		if err := ffmpeg.Start(); err != nil {
			fmt.Println(err)
			return
		}

		_ = drone.On(tello.FlightDataEvent, func(data interface{}) {
			// TODO: protect flight data from race condition
			// flightData = data.(*tello.FlightData)
		})

		drone.On(tello.ConnectedEvent, func(data interface{}) {
			fmt.Println("Connected")
			drone.StartVideo()
			drone.SetVideoEncoderRate(tello.VideoBitRateAuto)
			drone.SetExposure(0)

			gobot.Every(30*time.Millisecond, func() {
				drone.StartVideo()
			})
		})

		drone.On(tello.VideoFrameEvent, func(data interface{}) {
			pkt := data.([]byte)
			if _, err := ffmpegIn.Write(pkt); err != nil {
				fmt.Println(err)
			}
		})

		robot := gobot.NewRobot("tello",
			[]gobot.Connection{},
			[]gobot.Device{drone},
		)

		// calling Start(false) lets the Start routine return immediately without an additional blocking goroutine
		_ = robot.Start()
	}()

	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()
	xmlFile := "haarcascade_frontalface_default.xml"
	if !classifier.Load(xmlFile) {
		fmt.Printf("Error reading cascade file: %v\n", xmlFile)
		return
	}
	blue := color.RGBA{0, 0, 255, 0}
	// now handle video frames from ffmpeg stream in main thread, to be macOS/Windows friendly
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
		rects := classifier.DetectMultiScale(img)
		if len(rects) > 0 {
			biggest := image.Point{}
			var rect image.Rectangle
			for _, re := range rects {
				if re.Size().X > biggest.X && re.Size().Y > biggest.Y {
					biggest = re.Size()
					rect = re
				}
			}
			gocv.Rectangle(&img, rect, blue, 3)

			size := gocv.GetTextSize("Human", gocv.FontHersheyPlain, 1.2, 2)
			pt := image.Pt(rect.Min.X+(rect.Min.X/2)-(size.X/2), rect.Min.Y-2)
			gocv.PutText(&img, "Human", pt, gocv.FontHersheyPlain, 1.2, blue, 2)
		}
		window.IMShow(img)
		if window.WaitKey(10) >= 0 {
			break
		}
	}
}
