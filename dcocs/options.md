# Optios

def augment_image_generator(image_paths, output_size=(224, 224), debug=False):
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image '{image_path}' does not exist.")

        script_directory = os.path.dirname(os.path.abspath(__file__))
        temp_image_dir = os.path.join(script_directory, "temp_image_dir")
        output_directory = os.path.join(script_directory, "augmented_images")
        os.makedirs(temp_image_dir, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        shutil.copy(image_path, temp_image_dir)

        # Set up Augmentor pipeline
        p = Augmentor.Pipeline(source_directory=temp_image_dir, output_directory=output_directory)
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.4)
        p.flip_top_bottom(probability=0.4)
        p.rotate_random_90(probability=0.75)
        p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
        p.scale(probability=0.5, scale_factor=1.2)
        p.skew(probability=0.1, magnitude=0.5)
        p.random_color(probability=0.6, min_factor=0.6, max_factor=1.5)
        p.random_brightness(probability=0.6, min_factor=0.6, max_factor=1.5)
        p.random_contrast(probability=0.6, min_factor=0.6, max_factor=1.5)
        p.resize(probability=1, width=output_size[0], height=output_size[1], resample_filter="BICUBIC")

        if debug:
            p.sample(5, multi_threaded=False)
        else:
            p.sample(1, multi_threaded=False)

        augmented_image_path = os.path.join(output_directory, os.listdir(output_directory)[0])
        augmented_image = cv2.imread(augmented_image_path)
        
        if os.path.exists(augmented_image_path):
            os.remove(augmented_image_path)

        shutil.rmtree(temp_image_dir)

        augmented_image = tf.convert_to_tensor(augmented_image, dtype=tf.float32)
        yield augmented_image
