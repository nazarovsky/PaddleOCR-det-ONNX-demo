# PaddleOCR-det-ONNX-demo
Simple demo for converting PaddleOCR text detector model to ONNX and ONNX inference 

This simple demo shows how to export trained [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) detection model to ONNX

To export ONNX model add following lines after `model.eval()` in the `tools\export_model.py` in PaddleOCR repo
```
    model.eval()
###
    if config['Architecture']['model_type'] == "det":
        input_spec = paddle.static.InputSpec(shape=[1, 3, 736, 1280], dtype='float32', name='image')
        paddle.onnx.export(model, os.path.join(config['Global']['save_inference_dir'],'det_db'), input_spec=[input_spec], opset_version=10, enable_onnx_checker=True)
###
```
and run 
```
python3 tools/export_model.py -c configs/det/your_det_config.yml -o Global.pretrained_model=output/det_mv3/best_accuracy  Global.save_inference_dir=output/det_db/ Global.load_static_weights=false 
```

To test ONNX inference
```
python3 demo_razmetka.py
```





