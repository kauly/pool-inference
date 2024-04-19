import { useEffect, useState } from 'react'
import type { ChangeEvent } from 'react'

import { imageDataToTensor, process_output, runPoolModel } from './utils'
import defaultImage from './assets/FIMI0238.jpg'

const YOLO_IMAGE_SIZE = 640

const image = new Image()
const canvas = document.createElement('canvas')
canvas.width = YOLO_IMAGE_SIZE
canvas.height = YOLO_IMAGE_SIZE
interface ImageProps {
  url?: string
  imageData?: ImageData
}

const initialImageProps: ImageProps = {
  url: defaultImage,
  imageData: undefined,
}

function App() {
  const [loading, setLoading] = useState(false)
  const [imageProps, setImageProps] = useState<ImageProps>(initialImageProps)

  // draw the image on the canvas
  const handleImageUpload = async (ev: ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0]

    if (!file)
      return

    if (!canvas)
      return

    canvas.width = YOLO_IMAGE_SIZE
    canvas.height = YOLO_IMAGE_SIZE
    const ctx = canvas.getContext('2d')

    if (!ctx)
      return

    const url = URL.createObjectURL(file)

    if (imageProps.url)
      URL.revokeObjectURL(imageProps.url)

    image.src = url
    image.onload = () => {
      ctx.drawImage(image, 0, 0, YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE)
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      setImageProps({
        url,
        imageData: imgData,
      })
    }
  }

  const handleInference = async () => {
    try {
      if (!imageProps.url)
        throw new Error('No image found')

      setLoading(true)
      const pixels = imageProps.imageData?.data

      if (!pixels)
        throw new Error('No image data found')

      const preprocessedData = imageDataToTensor(pixels)
      const result = await runPoolModel(preprocessedData)
      // @ts-expect-error - result
      const boxes = process_output(result, YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE)

      if (!canvas)
        return
      const ctx = canvas.getContext('2d')
      if (!ctx)
        return
      ctx.strokeStyle = 'red'
      ctx.lineWidth = 3
      ctx.font = '20px Arial'
      boxes.forEach((box) => {
        const [x1, y1, x2, y2, objectType, probability] = box
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
        ctx.fillText(`${objectType} ${probability.toFixed(2)}`, x1, y1 - 5)
      })

      // get an image  from the canvas
      const url = canvas.toDataURL()
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)

      setImageProps({
        url,
        imageData: imgData,
      })
    }
    catch (error) {
      console.error(error)
    }
    finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const ctx = canvas.getContext('2d')
    if (!ctx)
      return
    image.src = defaultImage
    image.onload = () => {
      ctx.drawImage(image, 0, 0, YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE)
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      setImageProps({
        url: defaultImage,
        imageData: imgData,
      })
    }
  }, [])

  return (
    <div className="w-full bg-slate-800 ">
      <div className="h-screen container mx-auto flex max-w-4xl p-8">
        <div className="flex flex-col w-full space-y-8">
          <p className="text-4xl text-blue-400">YOLO pool detection</p>
          <div className="flex justify-between">
            <div>
              <label className="text-blue-400 pr-4" htmlFor="imgUploader">
                Upload an image:
              </label>
              <input
                type="file"
                accept="image/*"
                id="imgUploader"
                className="text-blue-50"
                onChange={handleImageUpload}
              />
            </div>
            <button onClick={handleInference} className="bg-blue-400 rounded-lg text-white p-2">
              {loading ? 'loading' : 'Detect'}
            </button>
          </div>
          <img className="aspect-video" src={imageProps.url} />

        </div>
      </div>
    </div>
  )
}

export default App
