from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

class DRIT(object) :
    def __init__(self, sess, args):
        self.model_name = 'DRIT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.num_attribute = args.num_attribute # for test
        self.guide_img = args.guide_img
        self.direction = args.direction

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.d_content_init_lr = args.lr / 2.5
        self.ch = args.ch
        self.concat = args.concat

        """ Weight """
        self.content_adv_w = args.content_adv_w
        self.domain_adv_w = args.domain_adv_w
        self.cycle_w = args.cycle_w
        self.recon_w = args.recon_w
        self.latent_w = args.latent_w
        self.kl_w = args.kl_w

        """ Generator """
        self.n_layer = args.n_layer
        self.n_z = args.n_z

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale
        self.n_d_con = args.n_d_con
        self.multi = True if args.n_scale > 1 else False
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# decay_flag : ", self.decay_flag)
        print("# epoch : ", self.epoch)
        print("# decay_epoch : ", self.decay_epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# attribute in test phase : ", self.num_attribute)

        print()

        print("##### Generator #####")
        print("# layer : ", self.n_layer)
        print("# z dimension : ", self.n_z)
        print("# concat : ", self.concat)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# multi-scale Dis : ", self.n_scale)
        print("# updating iteration of con_dis : ", self.n_d_con)
        print("# spectral_norm : ", self.sn)

        print()

        print("##### Weight #####")
        print("# domain_adv_weight : ", self.domain_adv_w)
        print("# content_adv_weight : ", self.content_adv_w)
        print("# cycle_weight : ", self.cycle_w)
        print("# recon_weight : ", self.recon_w)
        print("# latent_weight : ", self.latent_w)
        print("# kl_weight : ", self.kl_w)

    ##################################################################################
    # Encoder and Decoders
    ##################################################################################

    def content_encoder(self, x, is_training=True, reuse=False, scope='content_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            x = lrelu(x, 0.01)

            for i in range(2) :
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = relu(x)

                channel = channel * 2


            for i in range(1, self.n_layer) :
                x = resblock(x, channel, scope='resblock_'+str(i))

        with tf.variable_scope('content_encoder_share', reuse=tf.AUTO_REUSE) :
            x = resblock(x, channel, scope='resblock_share')
            x = gaussian_noise_layer(x, is_training)

            return x

    def attribute_encoder(self, x, reuse=False, scope='attribute_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            x = relu(x)
            channel = channel * 2

            x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_0')
            x = relu(x)
            channel = channel * 2


            for i in range(1, self.n_layer) :
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = relu(x)

            x = global_avg_pooling(x)
            x = conv(x, channels=self.n_z, kernel=1, stride=1, scope='attribute_logit')

            return x

    def attribute_encoder_concat(self, x, reuse=False, scope='attribute_encoder_concat'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv')

            for i in range(1, self.n_layer) :
                channel = channel * (i+1)
                x = basic_block(x, channel, scope='basic_block_' + str(i))

            x = lrelu(x, 0.2)
            x = global_avg_pooling(x)

            mean = fully_conneted(x, channels=self.n_z, scope='z_mean')
            logvar = fully_conneted(x, channels=self.n_z, scope='z_logvar')

            return mean, logvar

    def MLP(self, z, reuse=False, scope='MLP'):
        channel = self.ch * self.n_layer
        with tf.variable_scope(scope, reuse=reuse) :

            for i in range(2) :
                z = fully_conneted(z, channel, scope='fully_' + str(i))
                z = relu(z)

            z = fully_conneted(z, channel*self.n_layer, scope='fully_logit')

            return z

    def generator(self, x, z, reuse=False, scope="generator"):
        channel = self.ch * self.n_layer
        with tf.variable_scope(scope, reuse=reuse) :
            z = self.MLP(z, reuse=reuse)
            z = tf.split(z, num_or_size_splits=self.n_layer, axis=-1)

            for i in range(self.n_layer) :
                x = mis_resblock(x, z[i], channel, scope='mis_resblock_' + str(i))

            for i in range(2) :
                x = deconv(x, channel // 2, kernel=3, stride=2, scope='deconv_' + str(i))
                x = layer_norm(x, scope='layer_norm_' + str(i))
                x = relu(x)

                channel = channel // 2

            x = deconv(x, channels=self.img_ch, kernel=1, stride=1, scope='G_logit')
            x = tanh(x)

            return x

    def generator_concat(self, x, z, reuse=False, scope='generator_concat'):
        channel = self.ch * self.n_layer
        with tf.variable_scope('generator_concat_share', reuse=tf.AUTO_REUSE) :
            x = resblock(x, channel, scope='resblock')

        with tf.variable_scope(scope, reuse=reuse) :
            channel = channel + self.n_z
            x = expand_concat(x, z)

            for i in range(1, self.n_layer) :
                x = resblock(x, channel, scope='resblock_' + str(i))

            for i in range(2) :
                channel = channel + self.n_z
                x = expand_concat(x, z)

                x = deconv(x, channel // 2, kernel=3, stride=2, scope='deconv_' + str(i))
                x = layer_norm(x, scope='layer_norm_' + str(i))
                x = relu(x)

                channel = channel // 2

            x = expand_concat(x, z)
            x = deconv(x, channels=self.img_ch, kernel=1, stride=1, scope='G_logit')
            x = tanh(x)

            return x



    ##################################################################################
    # Discriminator
    ##################################################################################

    def content_discriminator(self, x, reuse=False, scope='content_discriminator'):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch * self.n_layer
            for i in range(3) :
                x = conv(x, channel, kernel=7, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x, 0.01)

            x = conv(x, channel, kernel=4, stride=1, scope='conv_3')
            x = lrelu(x, 0.01)

            x = conv(x, channels=1, kernel=1, stride=1, scope='D_content_logit')
            D_logit.append(x)

            return D_logit

    def multi_discriminator(self, x_init, reuse=False, scope="multi_discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            for scale in range(self.n_scale) :
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.01)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='ms_' + str(scale) +'conv_' + str(i))
                    x = lrelu(x, 0.01)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='ms_' + str(scale) + 'D_logit')
                D_logit.append(x)

                x_init = down_sample(x_init)

            return D_logit

    def discriminator(self, x, reuse=False, scope="discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch
            x = conv(x, channel, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv')
            x = lrelu(x, 0.01)

            for i in range(1, self.n_dis) :
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                x = lrelu(x, 0.01)

                channel = channel * 2

            x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='D_logit')
            D_logit.append(x)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def Encoder_A(self, x_A, is_training=True, random_fake=False, reuse=False):
        mean = None
        logvar = None

        content_A = self.content_encoder(x_A, is_training=is_training, reuse=reuse, scope='content_encoder_A')

        if self.concat :
            mean, logvar = self.attribute_encoder_concat(x_A, reuse=reuse, scope='attribute_encoder_concat_A')
            if random_fake :
                attribute_A = mean
            else :
                attribute_A = z_sample(mean, logvar)
        else :
            attribute_A = self.attribute_encoder(x_A, reuse=reuse, scope='attribute_encoder_A')

        return content_A, attribute_A, mean, logvar

    def Encoder_B(self, x_B, is_training=True, random_fake=False, reuse=False):
        mean = None
        logvar = None

        content_B = self.content_encoder(x_B, is_training=is_training, reuse=reuse, scope='content_encoder_B')

        if self.concat:
            mean, logvar = self.attribute_encoder_concat(x_B, reuse=reuse, scope='attribute_encoder_concat_B')
            if random_fake :
                attribute_B = mean

            else :
                attribute_B = z_sample(mean, logvar)
        else:
            attribute_B = self.attribute_encoder(x_B, reuse=reuse, scope='attribute_encoder_B')

        return content_B, attribute_B, mean, logvar

    def Decoder_A(self, content_B, attribute_A, reuse=False):
        # x = fake_A, identity_A, random_fake_A
        # x = (B, A), (A, A), (B, z)
        if self.concat :
            x = self.generator_concat(x=content_B, z=attribute_A, reuse=reuse, scope='generator_concat_A')
        else :
            x = self.generator(x=content_B, z=attribute_A, reuse=reuse, scope='generator_A')

        return x

    def Decoder_B(self, content_A, attribute_B, reuse=False):
        # x = fake_B, identity_B, random_fake_B
        # x = (A, B), (B, B), (A, z)
        if self.concat :
            x = self.generator_concat(x=content_A, z=attribute_B, reuse=reuse, scope='generator_concat_B')
        else :
            x = self.generator(x=content_A, z=attribute_B, reuse=reuse, scope='generator_B')

        return x

    def discriminate_real(self, x_A, x_B):
        if self.multi :
            real_A_logit = self.multi_discriminator(x_A, scope='multi_discriminator_A')
            real_B_logit = self.multi_discriminator(x_B, scope='multi_discriminator_B')

        else :
            real_A_logit = self.discriminator(x_A, scope="discriminator_A")
            real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        if self.multi :
            fake_A_logit = self.multi_discriminator(x_ba, reuse=True, scope='multi_discriminator_A')
            fake_B_logit = self.multi_discriminator(x_ab, reuse=True, scope='multi_discriminator_B')

        else :
            fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
            fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def discriminate_content(self, content_A, content_B, reuse=False):
        content_A_logit = self.content_discriminator(content_A, reuse=reuse, scope='content_discriminator')
        content_B_logit = self.content_discriminator(content_B, reuse=True, scope='content_discriminator')

        return content_A_logit, content_B_logit


    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='lr')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

        gpu_device = '/gpu:0'
        trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))


        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()


        self.domain_A = trainA_iterator.get_next()
        self.domain_B = trainB_iterator.get_next()


        """ Define Encoder, Generator, Discriminator """
        random_z = tf.random_normal(shape=[self.batch_size, self.n_z], mean=0.0, stddev=1.0, dtype=tf.float32)

        # encode
        content_a, attribute_a, mean_a, logvar_a = self.Encoder_A(self.domain_A)
        content_b, attribute_b, mean_b, logvar_b = self.Encoder_B(self.domain_B)

        # decode (fake, identity, random)
        fake_a = self.Decoder_A(content_B=content_b, attribute_A=attribute_a)
        fake_b = self.Decoder_B(content_A=content_a, attribute_B=attribute_b)

        recon_a = self.Decoder_A(content_B=content_a, attribute_A=attribute_a, reuse=True)
        recon_b = self.Decoder_B(content_A=content_b, attribute_B=attribute_b, reuse=True)

        random_fake_a = self.Decoder_A(content_B=content_b, attribute_A=random_z, reuse=True)
        random_fake_b = self.Decoder_B(content_A=content_a, attribute_B=random_z, reuse=True)

        # encode & decode again for cycle-consistency
        content_fake_a, attribute_fake_a, _, _ = self.Encoder_A(fake_a, reuse=True)
        content_fake_b, attribute_fake_b, _, _ = self.Encoder_B(fake_b, reuse=True)

        cycle_a = self.Decoder_A(content_B=content_fake_b, attribute_A=attribute_fake_a, reuse=True)
        cycle_b = self.Decoder_B(content_A=content_fake_a, attribute_B=attribute_fake_b, reuse=True)

        # for latent regression
        _, attribute_fake_random_a, _, _ = self.Encoder_A(random_fake_a, random_fake=True, reuse=True)
        _, attribute_fake_random_b, _, _ = self.Encoder_B(random_fake_b, random_fake=True, reuse=True)


        # discriminate
        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(fake_a, fake_b)
        random_fake_A_logit, random_fake_B_logit = self.discriminate_fake(random_fake_a, random_fake_b)
        content_A_logit, content_B_logit = self.discriminate_content(content_a, content_b)


        """ Define Loss """
        g_adv_loss_a = generator_loss(self.gan_type, fake_A_logit) + generator_loss(self.gan_type, random_fake_A_logit)
        g_adv_loss_b = generator_loss(self.gan_type, fake_B_logit) + generator_loss(self.gan_type, random_fake_B_logit)

        g_con_loss_a = generator_loss(self.gan_type, content_A_logit, content=True)
        g_con_loss_b = generator_loss(self.gan_type, content_B_logit, content=True)

        g_cyc_loss_a = L1_loss(cycle_a, self.domain_A)
        g_cyc_loss_b = L1_loss(cycle_b, self.domain_B)

        g_rec_loss_a = L1_loss(recon_a, self.domain_A)
        g_rec_loss_b = L1_loss(recon_b, self.domain_B)

        g_latent_loss_a = L1_loss(attribute_fake_random_a, random_z)
        g_latent_loss_b = L1_loss(attribute_fake_random_b, random_z)

        if self.concat :
            g_kl_loss_a = kl_loss(mean_a, logvar_a) + l2_regularize(content_a)
            g_kl_loss_b = kl_loss(mean_b, logvar_b) + l2_regularize(content_b)
        else :
            g_kl_loss_a = l2_regularize(attribute_a) + l2_regularize(content_a)
            g_kl_loss_b = l2_regularize(attribute_b) + l2_regularize(content_b)


        d_adv_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        d_adv_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)

        d_con_loss = discriminator_loss(self.gan_type, content_A_logit, content_B_logit)

        Generator_A_domain_loss = self.domain_adv_w * g_adv_loss_a
        Generator_A_content_loss = self.content_adv_w * g_con_loss_a
        Generator_A_cycle_loss = self.cycle_w * g_cyc_loss_b
        Generator_A_recon_loss = self.recon_w * g_rec_loss_a
        Generator_A_latent_loss = self.latent_w * g_latent_loss_a
        Generator_A_kl_loss = self.kl_w * g_kl_loss_a

        Generator_A_loss = Generator_A_domain_loss + \
                           Generator_A_content_loss + \
                           Generator_A_cycle_loss + \
                           Generator_A_recon_loss + \
                           Generator_A_latent_loss + \
                           Generator_A_kl_loss

        Generator_B_domain_loss = self.domain_adv_w * g_adv_loss_b
        Generator_B_content_loss = self.content_adv_w * g_con_loss_b
        Generator_B_cycle_loss = self.cycle_w * g_cyc_loss_a
        Generator_B_recon_loss = self.recon_w * g_rec_loss_b
        Generator_B_latent_loss = self.latent_w * g_latent_loss_b
        Generator_B_kl_loss = self.kl_w * g_kl_loss_b

        Generator_B_loss = Generator_B_domain_loss + \
                           Generator_B_content_loss + \
                           Generator_B_cycle_loss + \
                           Generator_B_recon_loss + \
                           Generator_B_latent_loss + \
                           Generator_B_kl_loss

        Discriminator_A_loss = self.domain_adv_w * d_adv_loss_a
        Discriminator_B_loss = self.domain_adv_w * d_adv_loss_b
        Discriminator_content_loss = self.content_adv_w * d_con_loss

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.Discriminator_content_loss = Discriminator_content_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'encoder' in var.name or 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name and 'content' not in var.name]
        D_content_vars = [var for var in t_vars if 'content_discriminator' in var.name]

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.Discriminator_content_loss, D_content_vars), clip_norm=5)

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)
        self.D_content_optim = tf.train.AdamOptimizer(self.d_content_init_lr, beta1=0.5, beta2=0.999).apply_gradients(zip(grads, D_content_vars))


        """" Summary """
        self.lr_write = tf.summary.scalar("learning_rate", self.lr)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_A_domain_loss = tf.summary.scalar("G_A_domain_loss", Generator_A_domain_loss)
        self.G_A_content_loss = tf.summary.scalar("G_A_content_loss", Generator_A_content_loss)
        self.G_A_cycle_loss = tf.summary.scalar("G_A_cycle_loss", Generator_A_cycle_loss)
        self.G_A_recon_loss = tf.summary.scalar("G_A_recon_loss", Generator_A_recon_loss)
        self.G_A_latent_loss = tf.summary.scalar("G_A_latent_loss", Generator_A_latent_loss)
        self.G_A_kl_loss = tf.summary.scalar("G_A_kl_loss", Generator_A_kl_loss)


        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.G_B_domain_loss = tf.summary.scalar("G_B_domain_loss", Generator_B_domain_loss)
        self.G_B_content_loss = tf.summary.scalar("G_B_content_loss", Generator_B_content_loss)
        self.G_B_cycle_loss = tf.summary.scalar("G_B_cycle_loss", Generator_B_cycle_loss)
        self.G_B_recon_loss = tf.summary.scalar("G_B_recon_loss", Generator_B_recon_loss)
        self.G_B_latent_loss = tf.summary.scalar("G_B_latent_loss", Generator_B_latent_loss)
        self.G_B_kl_loss = tf.summary.scalar("G_B_kl_loss", Generator_B_kl_loss)

        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss,
                                        self.G_A_domain_loss, self.G_A_content_loss,
                                        self.G_A_cycle_loss, self.G_A_recon_loss,
                                        self.G_A_latent_loss, self.G_A_kl_loss,

                                        self.G_B_loss,
                                        self.G_B_domain_loss, self.G_B_content_loss,
                                        self.G_B_cycle_loss, self.G_B_recon_loss,
                                        self.G_B_latent_loss, self.G_B_kl_loss,

                                        self.all_G_loss])

        self.D_loss = tf.summary.merge([self.D_A_loss,
                                        self.D_B_loss,
                                        self.all_D_loss])

        self.D_content_loss = tf.summary.scalar("Discriminator_content_loss", self.Discriminator_content_loss)



        """ Image """
        self.fake_A = random_fake_a
        self.fake_B = random_fake_b

        self.real_A = self.domain_A
        self.real_B = self.domain_B


        """ Test """
        self.test_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_image')
        self.test_random_z = tf.random_normal(shape=[1, self.n_z], mean=0.0, stddev=1.0, dtype=tf.float32)

        test_content_a, _, _, _ = self.Encoder_A(self.test_image, is_training=False, reuse=True)
        test_content_b, _, _, _ = self.Encoder_B(self.test_image, is_training=False, reuse=True)

        self.test_fake_A = self.Decoder_A(content_B=test_content_b, attribute_A=self.test_random_z, reuse=True)
        self.test_fake_B = self.Decoder_B(content_A=test_content_a, attribute_B=self.test_random_z, reuse=True)

        """ Guided Image Translation """
        self.content_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='content_image')
        self.attribute_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='guide_attribute_image')

        guide_content_A, guide_attribute_A, _, _ = self.Encoder_A(self.content_image, is_training=False, reuse=True)
        guide_content_B, guide_attribute_B, _, _ = self.Encoder_B(self.attribute_image, is_training=False, reuse=True)

        self.guide_fake_A = self.Decoder_A(content_B=guide_content_B, attribute_A=guide_attribute_A, reuse=True)
        self.guide_fake_B = self.Decoder_B(content_A=guide_content_A, attribute_B=guide_attribute_B, reuse=True)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag:
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)  # linear decay

            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr : lr
                }

                summary_str = self.sess.run(self.lr_write, feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update content D
                _, d_con_loss, summary_str = self.sess.run([self.D_content_optim, self.Discriminator_content_loss, self.D_content_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                if (counter - 1) % self.n_d_con == 0 :
                    # Update D
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                    # Update G
                    batch_A_images, batch_B_images, fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.real_A, self.real_B, self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                    print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_con_loss: %.8f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, self.iteration, time.time() - start_time, d_con_loss, d_loss, g_loss))

                else :
                    print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_con_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, d_con_loss))

                if np.mod(idx + 1, self.print_freq) == 0:
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    # save_images(batch_B_images, [self.batch_size, 1],
                    #             './{}/real_B_{}_{:03d}_{:05d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))

                    # save_images(fake_A, [self.batch_size, 1],
                    #             './{}/fake_A_{}_{:03d}_{:05d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))

                # display training status
                counter += 1

                if np.mod(idx+1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.concat :
            concat = "_concat"
        else :
            concat = ""

        if self.sn :
            sn = "_sn"
        else :
            sn = ""

        return "{}{}_{}_{}_{}layer_{}dis_{}scale_{}con{}".format(self.model_name, concat, self.dataset_name, self.gan_type,
                                                                 self.n_layer, self.n_dis, self.n_scale, self.n_d_con, sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_attribute) :
                image_path = os.path.join(self.result_dir, '{}_attribute{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_image : sample_image})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_attribute):
                image_path = os.path.join(self.result_dir, '{}_attribute{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")
        index.close()

    def guide_test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        attribute_file = np.asarray(load_test_data(self.guide_img, size=self.img_size))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir, 'guide')
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        if self.direction == 'a2b' :
            for sample_file in test_A_files:  # A -> B
                print('Processing A image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_B, feed_dict={self.content_image: sample_image, self.attribute_image : attribute_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")

        else :
            for sample_file in test_B_files:  # B -> A
                print('Processing B image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_A, feed_dict={self.content_image: sample_image, self.attribute_image : attribute_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")
        index.close()
