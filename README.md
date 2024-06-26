# Pool inference

I did this project to test inference on browser using [onnxruntime-web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) with wasm as backend. I'm using a YOLOv8 model trained to detect swimming pools. The code for data pre and post process was taken from this [amazing tutorial](https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e#javascript).

## Vite and WASM

The following steps are necessary to make *onnxruntime-web* work.

- Copy the WASM modules of the library to the public folder, I'm using [vite-plugin-static-copy](https://www.npmjs.com/package/vite-plugin-static-copy) to copy them. Code at: `vite.config.ts`
- Set the wasm path `ort.env.wasm.wasmPaths = '/'`. Code at: `src/utils`

## Demo

https://pool-inference.pages.dev/
