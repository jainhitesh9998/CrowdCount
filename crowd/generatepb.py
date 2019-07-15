import tensorflow as tf

meta_path = '/home/developer/Surveillance/Scripts/SecureIt_filter/weights/crowd/model.ckpt.meta' # Your .meta file
output_node_names = ['add_12']    # Output nodes

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('/home/developer/Surveillance/Scripts/SecureIt_filter/weights/crowd/'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('dense_crowd_count.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
