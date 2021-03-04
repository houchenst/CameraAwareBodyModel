import tensorflow.compat.v1 as tf


def print_parsed_record(args, record):
    n_bbox = 100000
    n_surf = 100000
    n_points = n_bbox + n_surf
    n_vert = 6890
    n_frames = 1
    # Parse parameters for global configurations.
    n_dims = args.n_dims
    data_dir = args.data_dir
    sample_bbox = args.sample_bbox
    sample_surf = args.sample_surf
    batch_size = args.batch_size
    subject = args.subject
    motion = args.motion
    n_parts = args.n_parts


    fs = tf.parse_single_example(
        record, 
        features={
              'point':
                  tf.FixedLenFeature([n_frames * n_points * n_dims],
                                     tf.float32),
              'label':
                  tf.FixedLenFeature([n_frames * n_points * 1], tf.float32),
              'vert':
                  tf.FixedLenFeature([n_frames * n_vert * n_dims], tf.float32),
              'weight':
                  tf.FixedLenFeature([n_frames * n_vert * n_parts], tf.float32),
              'transform':
                  tf.FixedLenFeature(
                      [n_frames * n_parts * (n_dims + 1) * (n_dims + 1)],
                      tf.float32),
              'joint':
                  tf.FixedLenFeature([n_frames * n_parts * n_dims], tf.float32),
              'name':
                  tf.FixedLenFeature([], tf.string),
          }
    )

    fs['point'] = tf.reshape(fs['point'], [n_frames, n_points, n_dims])
    fs['label'] = tf.reshape(fs['label'], [n_frames, n_points, 1])
    fs['vert'] = tf.reshape(fs['vert'], [n_frames, n_vert, n_dims])
    fs['weight'] = tf.reshape(fs['weight'], [n_frames, n_vert, n_parts])
    fs['transform'] = tf.reshape(fs['transform'],
                                [n_frames, n_parts, n_dims + 1, n_dims + 1])
    fs['joint'] = tf.reshape(fs['joint'], [n_frames, n_parts, n_dims])
    
    print(fs)



# if __name__ == "__main__":
