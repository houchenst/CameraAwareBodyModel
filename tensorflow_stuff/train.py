from torch import nn
import argparse
from test import print_parsed_record
import tensorflow.compat.v1 as tf
SAMPLE_RECORD = "C:\\Users\\Trevor\\Brown\\ivl-research\\data\\amass\\sample_dataset\\train-00-00-00000-of-00025.tfrecords"




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the neural body model")

    # Dataset Parameters
    parser.add_argument("--data_dir", required=True, type=str, help="Directory to load data from.")
    parser.add_argument("--sample_bbox", default=1024, type=int, help="Number of bbox samples.") #TODO: where do the bounding boxes come in?
    parser.add_argument("--sample_surf", default=1024, type=int, help="Number of surface samples.")
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size.")
    parser.add_argument("--motion", default=0, type=int, help="Index of the motion for evaluation.")
    parser.add_argument("--subject", default=0, type=int, help="Index of the subject for training.")

    # Model Parameters
    parser.add_argument("--n_parts", default=24, type=int, help="Number of parts.")  #Number of body parts presumably?
    parser.add_argument("--total_dim", default=960, type=int, help="Dimension of the latent vector (in total).")
    parser.add_argument("--shared_decoder", default=False, type=bool, help="Whether to use shared decoder.") #TODO: what does this do
    parser.add_argument("--soft_blend", default=5., type=float, help="The constant to blend parts.") # TODO: what does this do
    parser.add_argument("--projection", default=True, type=bool, help="Whether to use projected shaped features.") #TODO: what does this do
    parser.add_argument("--level_set", default=0.5, type=float, help="The value of the level set.")
    parser.add_argument("--n_dims", default=3, type=int, help="The dimension of the query points.") #TODO: Can they query without 3 dimensions (2 dimensions)?

    # Training Parameters
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--train_dir", required=True, type=str, help="Training directory.")
    parser.add_argument("--max_steps", default=200000, type=int, help="Number of optimization steps.")
    parser.add_argument("--save_every", default=5000, type=int, help="Number of steps to save checkpoint.")
    parser.add_argument("--summary_every", default=500, type=int, help="Number of steps to write summary.") #What is the summary?
    parser.add_argument("--label_w", default=0.5, type=float, help="Weight of labeled vertices loss.")
    parser.add_argument("--minimal_w", default=0.05, type=float, help="Weight of minimal loss.")
    parser.add_argument("--use_vert", default=True, type=bool, help="Whether to use vertices on the mesh for training.") #alternative would be sampling new points?
    parser.add_argument("--use_joint", default=True, type=bool, help="Whether to use joint-based transformation.") #TODO: what is the alternative?
    parser.add_argument("--sample_vert", default=2048, type=int, help="Number of vertex samples.")

    # TODO: Evaluation and Tracking arguments

    args = parser.parse_args()

    # with open(SAMPLE_RECORD, 'r') as record:
    #     record = record.read()
    
    files = [SAMPLE_RECORD]

    data_files = tf.gfile.Glob(files)
    filenames = tf.data.Dataset.list_files(SAMPLE_RECORD, shuffle=True)
    data = filenames.interleave(
        lambda x: tf.data.TFRecordDataset([x]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.map(
        lambda x: print_parsed_record(args, x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    



    # print_parsed_record(args, record)