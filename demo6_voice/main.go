package main

import (
	"time"

	"gobot.io/x/gobot"
	"gobot.io/x/gobot/platforms/dji/tello"
)

func main() {
	drone := tello.NewDriver("8888")

	work := func() {
		drone.TakeOff()

		gobot.After(8*time.Second, func() {
			drone.FrontFlip()
		})

		gobot.After(13*time.Second, func() {
			drone.BackFlip()
		})

		gobot.After(18*time.Second, func() {
			drone.Land()
		})
	}

	robot := gobot.NewRobot("tello",
		[]gobot.Connection{},
		[]gobot.Device{drone},
		work,
	)

	robot.Start()
}
