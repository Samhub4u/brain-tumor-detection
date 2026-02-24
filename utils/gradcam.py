"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet50.
Generates visual explanations of model predictions.
"""

import numpy as np
import cv2
import tensorflow as tf
import keras


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Generate a Grad-CAM heatmap for the given image.

    Args:
        model: Keras model
        img_array: Preprocessed image array of shape (1, 224, 224, 3)
        last_conv_layer_name: Name of the last conv layer (auto-detected if None)

    Returns:
        heatmap: Numpy array of shape (H, W) with values in [0, 1]
    """
    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            # Also check inside nested models (like ResNet50 base)
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, keras.layers.Conv2D):
                        last_conv_layer_name = sub_layer.name
                        break
                if last_conv_layer_name:
                    break

    if last_conv_layer_name is None:
        last_conv_layer_name = 'conv5_block3_3_conv'  # ResNet50 default

    # Try to find the layer - may be nested inside a sub-model
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Search inside sub-models
        for layer in model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    last_conv_layer = layer.get_layer(last_conv_layer_name)
                    # Build a model that outputs the conv layer output
                    grad_model = keras.models.Model(
                        inputs=model.input,
                        outputs=[layer.get_layer(last_conv_layer_name).output, model.output]
                    )
                    break
                except ValueError:
                    continue
        else:
            # Fallback: use the output of the base model (first functional model)
            for layer in model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 10:
                    grad_model = keras.models.Model(
                        inputs=model.input,
                        outputs=[layer.output, model.output]
                    )
                    break
            else:
                return np.zeros((224, 224), dtype=np.float32)

    if 'grad_model' not in dir():
        grad_model = keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return np.zeros((224, 224), dtype=np.float32)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_img: Original image as numpy array (H, W, 3), values in [0, 255]
        heatmap: Grad-CAM heatmap of shape (H', W') with values in [0, 1]
        alpha: Transparency of the heatmap overlay
        colormap: OpenCV colormap to use

    Returns:
        superimposed: Image with heatmap overlay, shape (H, W, 3), values in [0, 255]
    """
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed = np.uint8(heatmap_colored * alpha + original_img * (1 - alpha))

    return superimposed
