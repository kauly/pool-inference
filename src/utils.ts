import * as ort from 'onnxruntime-web'
import type { Tensor } from 'onnxruntime-web'

ort.env.wasm.wasmPaths = '/'

const yolo_classes = ['pool']

function imageDataToTensor(
  pixels: ImageData['data'],
  dims: number[] = [1, 3, 640, 640],
): Tensor {
  const [redArray, greenArray, blueArray]: [number[], number[], number[]] = [
    [],
    [],
    [],
  ]

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < pixels.length; i += 4) {
    redArray.push(pixels[i])
    greenArray.push(pixels[i + 1])
    blueArray.push(pixels[i + 2])
    // skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [640, 640, 3] -> [3, 640, 640] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray)

  // 4. convert to float32
  const l = transposedData.length // length, we need this for the loop

  // create the Float32Array size 3 * 640 * 640 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3])
  for (let i = 0; i < l; i++)
    float32Data[i] = transposedData[i] / 255.0 // convert to float

  // 5. create the tensor object from onnxruntime-web.
  const inputTensor = new ort.Tensor('float32', float32Data, dims)

  return inputTensor
}

async function runPoolModel(preprocessedData: Tensor) {
  try {
    const model = await ort.InferenceSession.create('/pool-model.onnx')
    const outputs = await model.run({ images: preprocessedData })
    const result = outputs.output0.data as Float32Array

    return result
  }
  catch (error) {
    console.error(error)
  }
}

/**
 * Function used to convert RAW output from YOLOv8 to an array of detected objects.
 * Each object contain the bounding box of this object, the type of object and the probability
 * @param output Raw output of YOLOv8 network
 * @param img_width Width of original image
 * @param img_height Height of original image
 * @returns Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
 */
function process_output(output: number[], img_width: number, img_height: number) {
  let boxes: (number)[][] = []
  for (let index = 0; index < 8400; index++) {
    const [class_id, prob] = [...Array(80).keys()]
      .map(col => [col, output[8400 * (col + 4) + index]])
      .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0])
    if (prob < 0.5)
      continue

    const label = yolo_classes[class_id]
    const xc = output[index]
    const yc = output[8400 + index]
    const w = output[2 * 8400 + index]
    const h = output[3 * 8400 + index]
    const x1 = (xc - w / 2) / 640 * img_width
    const y1 = (yc - h / 2) / 640 * img_height
    const x2 = (xc + w / 2) / 640 * img_width
    const y2 = (yc + h / 2) / 640 * img_height
    // @ts-expect-error - label is the only string in the array
    boxes.push([x1, y1, x2, y2, label, prob])
  }

  boxes = boxes.sort((box1, box2) => box2[5] - box1[5])
  const result = []
  while (boxes.length > 0) {
    result.push(boxes[0])
    boxes = boxes.filter(box => iou(boxes[0], box) < 0.7)
  }

  return result
}

/**
 * Function calculates "Intersection-over-union" coefficient for specified two boxes
 * https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
 * @param box1 First box in format: [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format: [x1,y1,x2,y2,object_class,probability]
 * @returns Intersection over union ratio as a float number
 */
function iou(box1: number[], box2: number[]) {
  return intersection(box1, box2) / union(box1, box2)
}

/**
 * Function calculates union area of two boxes.
 *     :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
 *     :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
 *     :return: Area of the boxes union as a float number
 * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
 * @returns Area of the boxes union as a float number
 */
function union(box1: number[], box2: number[]) {
  const [box1_x1, box1_y1, box1_x2, box1_y2] = box1
  const [box2_x1, box2_y1, box2_x2, box2_y2] = box2
  const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
  const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

  return box1_area + box2_area - intersection(box1, box2)
}

/**
 * Function calculates intersection area of two boxes
 * @param box1 First box in format [x1,y1,x2,y2,object_class,probability]
 * @param box2 Second box in format [x1,y1,x2,y2,object_class,probability]
 * @returns Area of intersection of the boxes as a float number
 */
function intersection(box1: number[], box2: number[]) {
  const [box1_x1, box1_y1, box1_x2, box1_y2] = box1
  const [box2_x1, box2_y1, box2_x2, box2_y2] = box2
  const x1 = Math.max(box1_x1, box2_x1)
  const y1 = Math.max(box1_y1, box2_y1)
  const x2 = Math.min(box1_x2, box2_x2)
  const y2 = Math.min(box1_y2, box2_y2)

  return (x2 - x1) * (y2 - y1)
}

export { imageDataToTensor, runPoolModel, process_output }
