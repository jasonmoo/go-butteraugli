package main

import (
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log"

	"fmt"
	"image"
	"os"

	butteraugli "github.com/jasonmoo/go-butteraugli"
)

func main() {

	log.SetFlags(0)

	if len(os.Args) != 3 {
		log.Fatalln("Usage:", os.Args[0], "<image1> <image2>")
	}

	file1, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatalf("Error opening %q: %s", os.Args[1], err)
	}
	defer file1.Close()

	file2, err := os.Open(os.Args[2])
	if err != nil {
		log.Fatalf("Error opening %q: %s", os.Args[2], err)
	}
	defer file2.Close()

	img1, _, err := image.Decode(file1)
	if err != nil {
		log.Fatalf("Error decoding %q: %s", os.Args[1], err)
	}

	img2, _, err := image.Decode(file2)
	if err != nil {
		log.Fatalf("Error decoding %q: %s", os.Args[2], err)
	}

	dv, err := butteraugli.CompareImages(img1, img2)
	if err != nil {
		log.Fatalf("Butteraugli error: %s", err)
	}

	fmt.Println(dv)

}
