import os
import tensorflow as tf
import tf2onnx
import onnx

def convert_to_onnx(model_path, output_path):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return

    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    
    print("Converting to ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    onnx.save(model_proto, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'spoilage_model.keras')
    output_path = os.path.join(base_dir, 'models', 'spoilage_model.onnx')
    
    convert_to_onnx(model_path, output_path)
