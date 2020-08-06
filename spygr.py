        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }

        activation_fn = (
            tf.nn.relu6 if model_options.use_bounded_activation else tf.nn.relu)

        def hw_flatten(x, transpose=False):
            if is_training:
                bs = 8
            else:
                bs = 1
            if transpose:
                return tf.transpose(tf.reshape(
                    x, shape=[bs, -1, x.shape[-1]]), perm=[0, 2, 1])
            else:
                return tf.reshape(x, shape=[bs, -1, x.shape[-1]])

        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm,
                padding='SAME',
                stride=1,
                reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                if is_training:
                    bs = 8
                else:
                    bs = 1

                features = slim.conv2d(features, 256, 1, scope='dim_to_256')
                inputs_shape = features.get_shape().as_list()
                M = 64

                # Conv: W
                m = slim.conv2d(features, M, 1, scope='dim_to_64')  # bs*h*w*M

                HWM = hw_flatten(m, transpose=False)    # bs*hw*M
                MHW = hw_flatten(m, transpose=True)     # bs*M*hw

                # Conv: Diag
                global_pool = slim.avg_pool2d(
                    features, [
                        inputs_shape[1], inputs_shape[2]], padding='VALID')     # bs*1*1*c

                global_pool = tf.matrix_diag(
                    tf.nn.softmax(slim.conv2d(global_pool, M, 1, scope='GP_dim_to_64')))    # bs*M*M*M

                global_pool = tf.reduce_mean(global_pool, axis=[1, 2])

                # Laplacia Matrix
                A = tf.matmul(tf.matmul(HWM, global_pool), MHW)     # bs*HW*HW
                D = tf.matrix_diag(tf.reduce_sum(A, 2))
                # d = tf.matrix_inverse(tf.sqrt(tf.matrix_diag(tf.reduce_sum(A, 2)), name=None))  # bs*HW*HW
                # L = tf.matmul(tf.matmul(d, A), d)   # bs*HW*HW
                L = tf.subtract(D, A)
                L_features = tf.matmul(
                    L, hw_flatten(
                        features, transpose=False))    # bs*HW*C
                L_features = tf.reshape(L_features,
                                        shape=[bs, inputs_shape[1], inputs_shape[2], inputs_shape[-1]])     # bs*H*W*C
                # spygr_features = tf.subtract(features, L_features)  # L = I -
                # D^(-1/2)*A*D^(-1/2)
                spygr_features = slim.conv2d(
                    L_features, inputs_shape[-1], 1, scope='get_spygr_features')
                # spygr_features = activation_fn(spygr_features)

                return tf.add_n([features, spygr_features],
                                name='spygr'), end_points
