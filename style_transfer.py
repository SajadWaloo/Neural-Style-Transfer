import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the VGG19 model with imagenet weights
vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(None, None, 3))

# Load content and style images
content_image_path = '/Users/mac/Desktop/Nasir/boat.jpg'
style_image_path = '/Users/mac/Desktop/Nasir/boat1.jpg'

# Load and preprocess the content image
content_image = tf.keras.preprocessing.image.load_img(content_image_path, target_size=(224, 224))
content_image = tf.keras.preprocessing.image.img_to_array(content_image)
content_image = tf.keras.applications.vgg19.preprocess_input(content_image)

# Load and preprocess the style image
style_image = tf.keras.preprocessing.image.load_img(style_image_path, target_size=(224, 224))
style_image = tf.keras.preprocessing.image.img_to_array(style_image)
style_image = tf.keras.applications.vgg19.preprocess_input(style_image)

# Display the content and style images
plt.subplot(1, 2, 1)
plt.imshow(tf.keras.preprocessing.image.array_to_img(content_image))
plt.title('Content Image')
plt.subplot(1, 2, 2)
plt.imshow(tf.keras.preprocessing.image.array_to_img(style_image))
plt.title('Style Image')
plt.show()

print('Content and style images loaded successfully.')

# Define the content loss function
def content_loss(content_features, generated_features):
    return tf.reduce_mean(tf.square(generated_features - content_features))

# Define the style loss function
def style_loss(style_features, generated_features):
    style_losses = []
    for style_feature, generated_feature in zip(style_features, generated_features):
        style_loss = tf.reduce_mean(tf.square(gram_matrix(style_feature) - gram_matrix(generated_feature)))
        style_losses.append(style_loss)
    return tf.reduce_mean(style_losses)

# Define the gram matrix calculation
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Extract features from intermediate layers of the VGG19 model
def extract_features(image, model):
    features = []
    for layer in model.layers:
        image = layer(image)
        if layer.name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
            features.append(image)
    return features

# Extract features from the content and style images
content_features = extract_features(content_image, vgg19)
style_features = extract_features(style_image, vgg19)

print('Content and style features extracted successfully.')

# Define the generated image as a trainable variable
generated_image = tf.Variable(content_image, dtype=tf.float32)

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Define the total loss function
def total_loss(content_features, style_features, generated_features, content_weight=1e4, style_weight=1e-2):
    content_loss_val = content_loss(content_features, generated_features)
    style_loss_val = style_loss(style_features, generated_features)
    return content_weight * content_loss_val + style_weight * style_loss_val

# Define the training step
@tf.function()
def train_step(generated_image, content_features, style_features):
    with tf.GradientTape() as tape:
        generated_features = extract_features(generated_image, vgg19)
        loss = total_loss(content_features, style_features, generated_features)
    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

# Training loop
num_iterations = 1000
for iteration in range(num_iterations):
    train_step(generated_image, content_features, style_features)
    if iteration % 100 == 0:
        print('Iteration:', iteration)

print('Model training completed.')

# Display the generated image
plt.imshow(tf.keras.preprocessing.image.array_to_img(generated_image))
plt.title('Generated Image')
plt.show()

print('Style transfer model executed successfully.')
