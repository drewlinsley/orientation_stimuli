'''
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
arxiv: https://arxiv.org/abs/1705.07750
'''
import numpy as np
import tensorflow as tf

from helper_classes.utils.layers import conv_batchnorm_relu
from helper_classes.utils.layers import maxpool
from helper_classes.utils.layers import upconv_2D
from tqdm import tqdm
import collections


NORMALIZATION_TYPE = 'instance'


def build_i3d(
        final_endpoint='Logits',
        use_batch_norm=False,
        ucrb=False,
        num_classes=101,
        rnn_type='fgru',
        #rnn_type='gru',
        dtype=tf.float32,
        spatial_squeeze=True,
        dropout_keep_prob=1.0,
        i3d_output=True,
        num_cores=8):
    """Build the gammanet."""
    print("*** Build sthgru ***")


    def model(inputs, is_training, fix_ff=True, timesteps=3, rnndim=8):
        """Init the gammanet hidden states then build the model."""
        net = inputs
        net_shape = net.get_shape().as_list()
        bs = net_shape[0]
        end_points, hidden_states = {}, collections.OrderedDict()
        # hidden_states = collections.OrderedDict()
        l4_shape = net_shape[2] // 2
        hidden_states['Mixed_4a_rnn'] = tf.zeros(
            [bs, net_shape[1], l4_shape, l4_shape, rnndim],
            dtype=dtype)
        # hidden_states['output'] = tf.zeros([bs, 1, 7, 7, 1024])
        if fix_ff:
            is_training_ff = False
        else:
            is_training_ff = is_training

        def rnn_layer(
                net,
                H,
                name,
                filters,
                default_3d,
                kernel_size,
                stride,
                is_training,
                num_cores,
                use_batch_norm,
                use_cross_replica_batch_norm,
                weight_scope,
                reuse,
                dtype,
                rnn_type='fgru',
                gate_kernel_size=(1, 1, 1),
                nl=tf.nn.relu):
            """Apply an rnn to net activities."""
            ucrb = use_cross_replica_batch_norm
            if rnn_type == 'gru':
                # Concat net + hidden state and get gates from a single conv
                cat_act = tf.concat([net, H], axis=-1)
                gates = conv_batchnorm_relu(
                    cat_act,
                    '%s_gru_gates' % name,
                    filters * 4,
                    default_3d=False,
                    activation=None,
                    kernel_size=gate_kernel_size,
                    stride=stride,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=ucrb,
                    weight_scope='weights',
                    reuse=reuse,
                    center=False,
                    scale=False,
                    share_scale=True,
                    share_center=True, preactivation=True,
                    normalization_type=NORMALIZATION_TYPE,
                    dtype=dtype,
                    gamma_initializer=tf.initializers.constant(0.1))
                splits = filters  # cat_act.get_shape().as_list()[-1] // 4
                in_gate_x = gates[..., :splits]
                out_gate_x = gates[..., splits: splits * 2]
                in_gate_h = gates[..., splits * 2: splits * 3]
                out_gate_h = gates[..., splits * 3:]

                in_gate = tf.nn.sigmoid(
                    in_gate_x + in_gate_h)
                out_gate = tf.nn.sigmoid(
                    out_gate_x + out_gate_h)
                gru_drive = tf.concat([net, H * in_gate], axis=-1)
                drive = conv_batchnorm_relu(
                    gru_drive,
                    '%s_gru_drive' % name,
                    filters * 2,
                    default_3d=False,
                    activation=None,
                    kernel_size=kernel_size,
                    stride=stride,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=ucrb,
                    weight_scope='weights',
                    gamma_initializer=tf.initializers.constant(0.1),
                    reuse=reuse,
                    center=False,
                    scale=False,
                    share_scale=True,
                    share_center=True, preactivation=True,
                    normalization_type=NORMALIZATION_TYPE,
                    dtype=dtype,
                    beta_initializer=tf.initializers.constant(0.))
                # splits = cat_act.get_shape().as_list()[-1] // 2
                ff_drive = drive[..., :splits]
                r_drive = drive[..., splits:]
                total_drive = nl(ff_drive + r_drive)
                H = (1 - out_gate) * H + out_gate * total_drive
            elif rnn_type == 'fgru':
                with tf.variable_scope('%s_modulators' % name, reuse=reuse):
                    alpha = tf.get_variable(
                        name='%s_alpha' % name,
                        shape=[1, 1, 1, 1, filters],
                        initializer=tf.constant_initializer(0.1),
                        dtype=dtype,
                        trainable=is_training)
                    mu = tf.get_variable(
                        name='%s_mu' % name,
                        shape=[1, 1, 1, 1, filters],
                        initializer=tf.constant_initializer(0.),
                        dtype=dtype,
                        trainable=is_training)
                    kappa = tf.get_variable(
                        name='%s_kappa' % name,
                        shape=[1, 1, 1, 1, filters],
                        initializer=tf.constant_initializer(0.5),
                        dtype=dtype,
                        trainable=is_training)
                    omega = tf.get_variable(
                        name='%s_omega' % name,
                        shape=[1, 1, 1, 1, filters],
                        initializer=tf.constant_initializer(0.5),
                        dtype=dtype,
                        trainable=is_training)

                # Circuit input  -- ADD OPTION FOR ATTENTION
                chronos = -np.log(
                    np.random.uniform(
                        low=1,
                        high=np.maximum(timesteps - 1, 1),
                        size=[filters]))
                input_gate = conv_batchnorm_relu(
                    H,
                    '%s_input_gate' % name,
                    filters,
                    default_3d=False,
                    activation=tf.nn.sigmoid,
                    kernel_size=gate_kernel_size,
                    stride=stride,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=ucrb,
                    weight_scope='weights',
                    reuse=reuse,
                    center=False,
                    scale=False,
                    share_scale=True,
                    share_center=True, preactivation=True,
                    normalization_type=NORMALIZATION_TYPE,
                    dtype=dtype,
                    bias_scope='%s_fgru_input_bias' % name,
                    gamma_initializer=tf.initializers.constant(0.1),
                    beta_initializer=tf.initializers.constant(chronos))
                    #print("input_gate, H: ",input_gate, H)
                print("input_gate, H: ",input_gate, H)

                input_drive = conv_batchnorm_relu(
                    #print("input_gate, H: ",input_gate, H)
                    input_gate * H,
                    '%s_S_drive' % name,
                    filters,
                    default_3d=False,
                    activation=None,
                    kernel_size=kernel_size,
                    stride=stride,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=ucrb,
                    center=False,
                    scale=False,
                    share_scale=True,
                    share_center=True, preactivation=True,
                    normalization_type=NORMALIZATION_TYPE,
                    dtype=dtype,
                    weight_scope='weights',
                    gamma_initializer=tf.initializers.constant(0.1),
                    beta_initializer=tf.initializers.constant(0.),
                    reuse=reuse)

                # Input integration
                S = tf.nn.relu(
                    net - tf.nn.relu((alpha * H + mu) * input_drive))

                # Circuit output
                chronos = np.concatenate((-chronos, np.zeros_like(chronos)))
                combo_drive = conv_batchnorm_relu(
                    S,
                    '%s_output_gate_and_F_drive' % name,
                    filters * 2,
                    default_3d=False,
                    activation=tf.nn.sigmoid,
                    kernel_size=gate_kernel_size,
                    stride=stride,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=ucrb,
                    weight_scope='weights',
                    reuse=reuse,
                    center=False,
                    scale=False,
                    share_scale=True,
                    share_center=True, preactivation=True,
                    normalization_type=NORMALIZATION_TYPE,
                    dtype=dtype,
                    bias_scope='%s_fgru_output_bias' % name,
                    gamma_initializer=tf.initializers.constant(0.1),
                    beta_initializer=tf.initializers.constant(chronos))
                output_gate = combo_drive[..., :filters]
                output_drive = combo_drive[..., filters:]

                # Output integration
                H_hat = tf.nn.relu(
                    kappa * (
                        S + output_drive) + omega * (S * output_drive))
                H = (output_gate * H) + (1 - output_gate) * H_hat
            elif rnn_type == 'debug':
                H = net
            else:
                raise NotImplementedError(rnn_type)
            return H  # , hidden_states

        def cond(
                inputs,
                is_training,
                end_points,
                reuse,
                idx,
                timesteps,
                Mixed_4a_rnn,
                Mixed_4b_rnn,
                Mixed_4c_rnn,
                Mixed_4d_rnn,
                Mixed_4e_rnn,
                Mixed_5a_rnn,
                Mixed_5b_rnn,
                Mixed_5c_rnn):
            """Cond for exiting while loop."""
            return idx < timesteps

        def topdown(
                is_training,
                reuse,
                use_batch_norm,
                ucrb,
                Mixed_4a_rnn,
                Mixed_4b_rnn,
                Mixed_4c_rnn,
                Mixed_4d_rnn,
                Mixed_4e_rnn,
                Mixed_5a_rnn,
                Mixed_5b_rnn,
                Mixed_5c_rnn,
                # td_list=[  # Full set of TDs
                #     ['Mixed_3a_rnn', 'Mixed_3b_rnn', 192],
                #     ['Mixed_3b_rnn', 'Mixed_4a_rnn', 46],
                #     ['Mixed_4a_rnn', 'Mixed_4b_rnn', 77],
                #     ['Mixed_4b_rnn', 'Mixed_4c_rnn', 82],
                #     ['Mixed_4c_rnn', 'Mixed_4d_rnn', 82],
                #     ['Mixed_4d_rnn', 'Mixed_4e_rnn', 82],
                #     ['Mixed_4e_rnn', 'Mixed_5a_rnn', 84],
                #     ['Mixed_5a_rnn', 'Mixed_5b_rnn', 123],
                #     ['Mixed_5b_rnn', 'Mixed_5c_rnn', 123],
                # ],
                td_list=[  # Partial set of TDs
                    ['Mixed_4a_rnn', 'Mixed_4e_rnn', 128],
                    ['Mixed_4e_rnn', 'Mixed_5a_rnn', 128],
                    ['Mixed_5a_rnn', 'Mixed_5c_rnn', 256],


                ],
                upsample=False,
                horizontal_kernel_size=(1, 1, 1),
                gate_kernel_size=(2, 1, 1),
                debug=False):
            """Topdown pass through the gammanet.

            Updates all hidden states in reverse order by applying
            the function:

            H^{(\ell)} = fGRU(Z=H^{(\ell)}, H=upsample(H^{(\ell + 1)}))

            Order fixed for inception.
            Mixed_3b_rnn
            Mixed_4e_rnn ->
            Mixed_5c_rnn ->

            TODO: first/last hidden states. Alternatively, all hidden states.
            """
            raise NotImplementedError("No top-down implemented at the moment.")
            if debug:
                use_batch_norm, ucrb = False, False
            assert len(td_list), 'No top-down layers in the list.'
            for tdl in reversed(td_list):
                h_name, td_name, h_features = tdl
                # Grab H and TD recurrent states
                if h_name == 'Mixed_4a_rnn':
                    h_act = Mixed_4a_rnn
                elif h_name == 'Mixed_4b_rnn':
                    h_act = Mixed_4b_rnn
                elif h_name == 'Mixed_4c_rnn':
                    h_act = Mixed_4c_rnn
                elif h_name == 'Mixed_4d_rnn':
                    h_act = Mixed_4d_rnn
                elif h_name == 'Mixed_4e_rnn':
                    h_act = Mixed_4e_rnn
                elif h_name == 'Mixed_5a_rnn':
                    h_act = Mixed_5a_rnn
                elif h_name == 'Mixed_5b_rnn':
                    h_act = Mixed_5b_rnn
                else:
                    raise NotImplementedError(h_name)

                if td_name == 'Mixed_4b_rnn':
                    td_act = Mixed_4b_rnn
                elif td_name == 'Mixed_4c_rnn':
                    td_act = Mixed_4c_rnn
                elif td_name == 'Mixed_4d_rnn':
                    td_act = Mixed_4d_rnn
                elif td_name == 'Mixed_4e_rnn':
                    td_act = Mixed_4e_rnn
                elif td_name == 'Mixed_5a_rnn':
                    td_act = Mixed_5a_rnn
                elif td_name == 'Mixed_5b_rnn':
                    td_act = Mixed_5b_rnn
                elif td_name == 'Mixed_5c_rnn':
                    td_act = Mixed_5c_rnn
                else:
                    raise NotImplementedError(h_name)

                # Apply TD
                with tf.variable_scope('%s_topdown' % h_name):
                    td_shape = td_act.get_shape().as_list()
                    h_shape = h_act.get_shape().as_list()
                    if td_shape[2] != h_shape[2] and td_shape[3] != h_shape[3]:
                        sample_factor = h_shape[2] // td_shape[2]
                        squeeze_td_act = tf.squeeze(td_act, 1)
                        if upsample:
                            resized_td_act = upconv_2D(
                                input_var=squeeze_td_act,
                                layer_name='%s_topdown_upsample' % h_name,
                                n_filters=td_shape[-1],
                                training=is_training,
                                kernel_size=(sample_factor, sample_factor),
                                strides=(sample_factor, sample_factor),
                                use_bias=False,
                                activation=tf.nn.relu,
                                center=False,
                                scale=False,
                                share_scale=True,
                                share_center=True, preactivation=True,
                                use_cross_replica_batch_norm=ucrb,
                                use_batch_norm=use_batch_norm,
                                dtype=dtype,
                                reuse=reuse,
                                normalization_type=NORMALIZATION_TYPE)
                        else:
                            resized_td_act = tf.image.resize_bilinear(
                                images=squeeze_td_act,
                                size=h_shape[2:4],
                                align_corners=True,
                                name='%s_topdown_upsample' % h_name)
                        resized_td_act = tf.expand_dims(resized_td_act, axis=1)
                    else:
                        resized_td_act = td_act
                    resized_td_act = conv_batchnorm_relu(
                        resized_td_act,
                        '%s_topdown_conv' % h_name,
                        h_features,
                        default_3d=False,
                        kernel_size=1,
                        stride=1,
                        center=False,
                        scale=False,
                        share_scale=True,
                        share_center=True, preactivation=True,
                        normalization_type=NORMALIZATION_TYPE,
                        is_training=is_training,
                        num_cores=num_cores,
                        use_batch_norm=use_batch_norm,
                        use_cross_replica_batch_norm=ucrb,
                        weight_scope='weights',
                        dtype=dtype,
                        reuse=reuse)
                    updated_h_act = rnn_layer(
                        net=h_act,
                        H=resized_td_act,
                        name='%s_rnn_topdown' % h_name,
                        filters=h_features,
                        default_3d=False,
                        kernel_size=horizontal_kernel_size,
                        gate_kernel_size=gate_kernel_size,
                        stride=(1, 1, 1),
                        is_training=is_training,
                        num_cores=num_cores,
                        rnn_type=rnn_type,
                        use_batch_norm=use_batch_norm,
                        use_cross_replica_batch_norm=ucrb,
                        weight_scope='weights',
                        dtype=dtype,
                        reuse=reuse)
                # Reassign TD interactions
                # Grab H and TD recurrent states
                if h_name == 'Mixed_4a_rnn':
                    Mixed_4a_rnn = updated_h_act
                elif h_name == 'Mixed_4b_rnn':
                    Mixed_4b_rnn = updated_h_act
                elif h_name == 'Mixed_4c_rnn':
                    Mixed_4c_rnn = updated_h_act
                elif h_name == 'Mixed_4d_rnn':
                    Mixed_4d_rnn = updated_h_act
                elif h_name == 'Mixed_4e_rnn':
                    Mixed_4e_rnn = updated_h_act
                elif h_name == 'Mixed_5a_rnn':
                    Mixed_5a_rnn = updated_h_act
                elif h_name == 'Mixed_5b_rnn':
                    Mixed_5b_rnn = updated_h_act
                else:
                    raise NotImplementedError(h_name)
            return [
                Mixed_4a_rnn,
                Mixed_4b_rnn,
                Mixed_4c_rnn,
                Mixed_4d_rnn,
                Mixed_4e_rnn,
                Mixed_5a_rnn,
                Mixed_5b_rnn,
                Mixed_5c_rnn]

        def horizontal(
                inputs,
                is_training,
                end_points,
                reuse,
                idx,
                timesteps,
                use_batch_norm,
                ucrb,
                Mixed_4a_rnn,
                horizontal_kernel_size=(3, 5, 5),   # Time/Height/Width  (try 255, 555)
                gate_kernel_size=(3, 1, 1),   # Time/Height/Widthi  (try 211, 511)
                debug=False):
            """
            Updates all hidden states in forward order by applying
            the function:

            H^{(\ell)} = fGRU(Z=CONV(activity^{(\ell + 1)}), H=H^{(\ell)}))

            Order fixed for inception.
            """
            if debug:
                use_batch_norm, ucrb = False, False
            Mixed_4a_rnn = rnn_layer(
                net=inputs,
                H=Mixed_4a_rnn,
                name='%s_rnn' % 'Mixed_4a',
                filters=net.get_shape().as_list()[-1],
                default_3d=False,
                kernel_size=horizontal_kernel_size,
                gate_kernel_size=gate_kernel_size,
                stride=(1, 1, 1),
                is_training=is_training,
                num_cores=num_cores,
                rnn_type=rnn_type,
                use_batch_norm=use_batch_norm,
                dtype=dtype,
                use_cross_replica_batch_norm=ucrb,
                weight_scope='weights', reuse=reuse)
            return Mixed_4a_rnn

        # Pre RNN ops
        end_point = 'Conv2d_1a_7x7'
        net = conv_batchnorm_relu(
            inputs,
            end_point, 64,
            default_3d=False,
            activation=None,
            kernel_size=(1, 7, 7),
            stride=(1, 1, 1),
            is_training=is_training_ff,
            num_cores=num_cores,
            use_batch_norm=use_batch_norm,
            normalization_type=NORMALIZATION_TYPE,
            dtype=dtype,
            use_cross_replica_batch_norm=ucrb,
            weight_scope='weights')
        # get_shape = net.get_shape().as_list()
        # print('{} : {}'.format(end_point, get_shape))

        print('Inputs: {}'.format(inputs.get_shape().as_list()))

        # 1x3x3 Max-pool, stride 1, 2, 2
        end_point = 'MaxPool2d_1_1'
        net = maxpool(
            net,  # Girik, consider changing the T dim to 2. Also consider removing this entire pool.
            end_point,
            ksize=[1, 1, 3, 3, 1],
            strides=[1, 1, 2, 2, 1], padding='SAME')
        get_shape = net.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))
        print("timesteps: ", timesteps)

        # 2nd FF layer
        # 1x1x1 Conv, stride 1
        end_point = 'Conv2d_2b_1x1'
        net = conv_batchnorm_relu(net, end_point, 64, default_3d=False,
            kernel_size=1, stride=1, is_training=is_training_ff, num_cores=num_cores,
             use_batch_norm=use_batch_norm, dtype=dtype, use_cross_replica_batch_norm=use_cross_replica_batch_norm, weight_scope='weights')
        get_shape = net.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))

        # 3x3x3 Conv, stride 1
        end_point = 'Conv2d_2c_3x3'
        net = conv_batchnorm_relu(net, end_point, 192, default_3d=False,
             kernel_size=(1, 3, 3), stride=1, is_training=is_training_ff, num_cores=num_cores,
             use_batch_norm=use_batch_norm, dtype=dtype, use_cross_replica_batch_norm=use_cross_replica_batch_norm, weight_scope='weights')
        get_shape = net.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))

        # 1x3x3 Max-pool, stride 1, 2, 2
        end_point = 'MaxPool3d_3a_3x3'
        net = maxpool(
            net,
            end_point,
            ksize=[1, 1, 3, 3, 1],
            strides=[1, 1, 2, 2, 1], padding='SAME')
        get_shape = net.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))

        # 1D conv, no normalization, rectification, -> rnndims
        net = conv_batchnorm_relu(
            net,
            'Conv_downsample',
            rnndim,
            default_3d=False,
            kernel_size=1,
            stride=1,
            is_training=is_training,
            num_cores=num_cores,
            use_batch_norm=False,  # use_batch_norm,
            normalization_type=NORMALIZATION_TYPE,
            dtype=dtype,
            use_cross_replica_batch_norm=False,  # ucrb,
            weight_scope='weights')

        # Run recurrent loop
        if is_training:
            reuse = False
        else:
            reuse = tf.AUTO_REUSE
        # idx = tf.constant(0)
        for idx in tqdm(
                range(timesteps),
                total=timesteps,
                desc='Building hgru timestep'):
            # Bottom-up then Top-down pass
            Mixed_4a_rnn = horizontal(
                inputs=net,
                is_training=is_training,
                end_points=end_points,
                reuse=reuse,
                idx=idx,
                timesteps=timesteps,
                use_batch_norm=use_batch_norm,
                ucrb=ucrb,
                Mixed_4a_rnn=hidden_states['Mixed_4a_rnn'])
            hidden_states['Mixed_4a_rnn'] = Mixed_4a_rnn
            # else:
            #     print('Skipping TD on final iteration.')
            if idx == 0:
                num_variables = tf.trainable_variables()
            else:
                # Make sure loop is not expanding
                new_num_variables = tf.trainable_variables()
                if len(num_variables) != len(new_num_variables):
                    num_variables = set(
                        [x.name for x in num_variables])
                    new_num_variables = set(
                        [x.name for x in new_num_variables])
                    raise RuntimeError(
                        num_variables.difference(new_num_variables))
            reuse = tf.AUTO_REUSE

        # Logits
        net = conv_batchnorm_relu(
            Mixed_4a_rnn,
            'Rnn_output_remap_normalization',
            Mixed_4a_rnn.get_shape().as_list()[-1],
            default_3d=False,
            kernel_size=1,
            stride=1,
            is_training=is_training,
            center=False,
            scale=False,
            share_scale=True,
            share_center=True,
            preactivation=True,
            normalization_type=NORMALIZATION_TYPE,
            num_cores=num_cores,
            use_batch_norm=use_batch_norm,
            dtype=dtype,
            use_cross_replica_batch_norm=ucrb,
            weight_scope='weights',
            reuse=reuse)

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            # 2x7x7 Average-pool, stride 1, 1, 1
            # net = avgpool(
            #     net, ksize=[1, 1, 7, 7, 1],
            #     strides=[1, 1, 1, 1, 1], padding='VALID')
            # get_shape = net.get_shape().as_list()
            # print('{} / Average-pool3D: {}'.format(end_point, get_shape))
            # end_points[end_point + '_average_pool3d'] = net
            net = tf.reduce_mean(net, reduction_indices=[2, 3], keep_dims=True)

            """
            # Try moving the dropout here instead
            net = conv_batchnorm_relu(
                net,
                'time_readout',
                1,  # Map time to 1 dimension
                kernel_size=[net.get_shape().as_list()[1], 1, 1],
                stride=1,
                activation=None,
                use_batch_norm=False,
                use_cross_replica_batch_norm=False,
                weight_scope='weights',
                is_training=is_training,
                dtype=dtype,
                normalization_type=NORMALIZATION_TYPE,
                num_cores=num_cores,
                default_3d=False)
            """
            # Dropout
            net = tf.nn.dropout(net, dropout_keep_prob)
            # 1x1x1 Conv, stride 1
            ns = net.get_shape().as_list()
            net = tf.reshape(net, [ns[0], 1, 1, 1, -1])
            logits = conv_batchnorm_relu(
                net,
                'Conv2d_0c_1x1',
                num_classes,
                kernel_size=1,
                stride=1,
                activation=None,
                use_batch_norm=use_batch_norm,
                use_cross_replica_batch_norm=ucrb,
                weight_scope='weights',
                is_training=is_training,
                dtype=dtype,
                normalization_type=NORMALIZATION_TYPE,
                num_cores=num_cores,
                default_3d=False)
            # get_shape = logits.get_shape().as_list()
            # print('{} / Conv2d_0c_1x1 : {}'.format(end_point, get_shape))
            if spatial_squeeze:
                # Removes dimensions of size 1 from the shape of a tensor
                # Specify which dimensions have to be removed: 2 and 3
                logits = tf.squeeze(logits, [1, 2, 3], name='SpatialSqueeze')
                # get_shape = logits.get_shape().as_list()
        return logits, idx

    return model


def InceptionI3d(
        final_endpoint='Logits',
        use_batch_norm=False,
        use_cross_replica_batch_norm=False,
        num_classes=None,
        spatial_squeeze=True,
        num_cores=8,
        dtype=tf.float32,
        i3d_output=True,
        dropout_keep_prob=1.0):
    assert num_classes is not None, 'Need to specify num_classes.'
    return build_i3d(
        final_endpoint=final_endpoint,
        use_batch_norm=use_batch_norm,
        ucrb=use_cross_replica_batch_norm,
        num_cores=num_cores,
        num_classes=num_classes,
        spatial_squeeze=spatial_squeeze,
        dtype=dtype,
        i3d_output=i3d_output,
        dropout_keep_prob=dropout_keep_prob)
