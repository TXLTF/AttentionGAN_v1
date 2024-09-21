import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
# from scipy.misc import imresize

from PIL import Image

# 调整图像大小


def resize_image(image, new_size, resample=Image.BICUBIC):
    """调整图像大小

    参数:
        image (PIL.Image.Image) -- 要调整大小的图像
        new_size (tuple)         -- 新图像的大小
        resample (int)           -- 重采样方法

    返回:
        PIL.Image.Image -- 调整大小后的图像
    """
    return image.resize(new_size, resample=resample)


# 如果Python版本是2，则使用Exception，否则使用ConnectionError
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

# 保存图像到磁盘


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """保存图像到磁盘。

    参数:
        webpage (the HTML class) -- 存储这些图像的HTML网页类（请参阅html.py以获取更多详细信息）
        visuals (OrderedDict)    -- 一个有序字典，存储（名称，图像（张量或numpy））对
        image_path (str)         -- 用于创建图像路径的字符串
        aspect_ratio (float)     -- 保存图像的纵横比
        width (int)              -- 图像将调整为宽度 x 宽度

    此函数将存储在'visuals'中的图像保存到'webpage'指定的HTML文件中。
    """
    # 获取图像目录
    image_dir = webpage.get_image_dir()
    # 获取图像路径的短路径
    short_path = ntpath.basename(image_path[0])
    # 获取图像名称
    name = os.path.splitext(short_path)[0]
    # 添加标题
    webpage.add_header(name)
    # 初始化图像、文本和链接列表
    ims, txts, links = [], [], []

    # 遍历visuals中的每个图像
    for label, im_data in visuals.items():
        # 将张量转换为图像
        im = util.tensor2im(im_data)
        # 创建图像名称
        image_name = '%s_%s.png' % (name, label)
        # 创建保存路径
        save_path = os.path.join(image_dir, image_name)
        # 获取图像的高度和宽度
        h, w, _ = im.shape
        # if aspect_ratio > 1.0:
        #     im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        # if aspect_ratio < 1.0:
        #     im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        # 如果纵横比大于1.0，则调整图像宽度
        if aspect_ratio > 1.0:
            new_width = int(w * aspect_ratio)
            im = resize_image(im, (h, new_width))
        # 如果纵横比小于1.0，则调整图像高度
        elif aspect_ratio < 1.0:
            new_height = int(h / aspect_ratio)
            im = resize_image(im, (new_height, w))
        # 保存图像
        util.save_image(im, save_path)

        # 将图像名称添加到列表中
        ims.append(image_name)
        # 将标签添加到文本列表中
        txts.append(label)
        # 将图像名称添加到链接列表中
        links.append(image_name)
    # 将图像添加到网页中
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """这个类包含几个函数，可以显示/保存图像并打印/保存日志信息。

    它使用Python库'visdom'进行显示，并使用Python库'dominate'（包装在'HTML'中）创建包含图像的HTML文件。
    """

    def __init__(self, opt):
        """初始化Visualizer类

        Parameters:
            opt -- 存储所有实验标志；需要是BaseOptions的子类
        Step 1: 缓存训练/测试选项
        Step 2: 连接到visdom服务器
        Step 3: 创建一个HTML对象以保存HTML过滤器
        Step 4: 创建一个日志文件以存储训练损失
        """
        self.opt = opt  # 缓存选项
        # 显示ID
        self.display_id = opt.display_id
        # 是否使用HTML
        self.use_html = opt.isTrain and not opt.no_html
        # 窗口大小
        self.win_size = opt.display_winsize
        # 名称
        self.name = opt.name
        # 端口
        self.port = opt.display_port
        # 是否保存
        self.saved = False
        # 是否连接到visdom服务器
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            # 列数
            self.ncols = opt.display_ncols
            # 连接到visdom服务器
            self.vis = visdom.Visdom(
                server=opt.display_server, port=opt.display_port, env=opt.display_env)
            # 检查连接
            if not self.vis.check_connection():
                self.create_visdom_connections()

        # 如果使用HTML
        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            # 创建web目录
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            # 创建images目录
            self.img_dir = os.path.join(self.web_dir, 'images')
            # 打印创建的web目录
            print('create web directory %s...' % self.web_dir)
            # 创建目录
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(
            opt.checkpoints_dir, opt.name, 'loss_log.txt')
        # 打开日志文件
        with open(self.log_name, "a") as log_file:
            # 获取当前时间
            now = time.strftime("%c")
            # 写入日志
            log_file.write(
                '================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """重置self.saved状态"""
        self.saved = False

    def create_visdom_connections(self):
        """如果程序无法连接到Visdom服务器，此函数将在端口< self.port >处启动新服务器"""
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\n无法连接到Visdom服务器。\n 尝试启动服务器....')
        print('命令: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """在visdom上显示当前结果；将当前结果保存到HTML文件。

        Parameters:
            visuals (OrderedDict) - - 要显示或保存的图像字典
            epoch (int) - - 当前epoch
            save_result (bool) - - 如果保存当前结果到HTML文件
        """
        # 在浏览器中使用visdom显示图像
        if self.display_id > 0:  # show images in the browser using visdom
            # 列数
            ncols = self.ncols
            # 如果列数大于0，则显示所有图像
            if ncols > 0:        # show all the images in one visdom panel
                # 列数
                ncols = min(ncols, len(visuals))
                # 获取图像的高度和宽度
                h, w = next(iter(visuals.values())).shape[:2]
                # 创建表格CSS
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # 创建一个图像表。
                title = self.name
                # 创建标签HTML
                label_html = ''
                # 创建标签HTML行
                label_html_row = ''
                # 创建图像列表
                images = []
                # 创建索引
                idx = 0
                # 遍历visuals中的每个图像
                for label, image in visuals.items():
                    # 将张量转换为图像
                    image_numpy = util.tensor2im(image)
                    # 将标签添加到标签HTML行中
                    label_html_row += '<td>%s</td>' % label
                    # 将图像添加到图像列表中
                    images.append(image_numpy.transpose([2, 0, 1]))
                    # 增加索引
                    idx += 1
                    # 如果idx能被ncols整除，则添加标签HTML行
                    if idx % ncols == 0:
                        # 将标签HTML行添加到标签HTML中
                        label_html += '<tr>%s</tr>' % label_html_row
                        # 重置标签HTML行
                        label_html_row = ''
                # 创建一个白色图像
                white_image = np.ones_like(
                    image_numpy.transpose([2, 0, 1])) * 255
                # 如果idx不能被ncols整除，则添加白色图像
                while idx % ncols != 0:
                    # 将白色图像添加到图像列表中
                    images.append(white_image)
                    # 将空标签添加到标签HTML行中
                    label_html_row += '<td></td>'
                    # 增加索引
                    idx += 1
                # 如果标签HTML行不为空，则添加标签HTML行
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # 尝试在visdom中显示图像
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    # 将标签HTML添加到visdom中
                    label_html = '<table>%s</table>' % label_html
                    # 将标签HTML添加到visdom中
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    # 如果无法连接到visdom服务器，则创建一个新的连接
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                # 如果列数为0，则每个图像显示在一个单独的visdom面板中
                idx = 1
                try:
                    # 遍历visuals中的每个图像
                    for label, image in visuals.items():
                        # 将张量转换为图像
                        image_numpy = util.tensor2im(image)
                        # 将图像添加到visdom中
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        # save images to an HTML file if they haven't been saved.
        # 如果使用HTML且未保存图像，则保存图像到HTML文件
        if self.use_html and (save_result or not self.saved):
            self.saved = True
            # 保存图像到磁盘
            for label, image in visuals.items():
                # 将张量转换为图像
                image_numpy = util.tensor2im(image)
                # 创建图像路径
                img_path = os.path.join(
                    self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                # 保存图像
                util.save_image(image_numpy, img_path)

            # 更新网站
            # 创建HTML对象
            webpage = html.HTML(
                self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            # 遍历epoch
            for n in range(epoch, 0, -1):
                # 添加标题
                webpage.add_header('epoch [%d]' % n)
                # 创建图像列表、文本列表和链接列表
                ims, txts, links = [], [], []
                # 遍历visuals中的每个图像
                for label, image_numpy in visuals.items():
                    # 将张量转换为图像
                    image_numpy = util.tensor2im(image)
                    # 创建图像路径
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    # 将图像路径添加到图像列表中
                    ims.append(img_path)
                    # 将标签添加到文本列表中
                    txts.append(label)
                    # 将图像路径添加到链接列表中
                    links.append(img_path)
                # 将图像添加到网页中
                webpage.add_images(ims, txts, links, width=self.win_size)
            # 保存网页
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """在visdom上显示当前损失；字典错误标签和值

        Parameters:
            epoch (int)           -- 当前epoch
            counter_ratio (float) -- 当前epoch的进度（百分比），介于0到1之间
            losses (OrderedDict)  -- 训练损失存储在（名称，浮点数）对的格式中
        """
        # 如果plot_data不存在，则创建plot_data
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        # 将当前epoch的进度添加到plot_data中
        self.plot_data['X'].append(epoch + counter_ratio)
        # 将当前epoch的损失添加到plot_data中
        self.plot_data['Y'].append([losses[k]
                                   for k in self.plot_data['legend']])
        try:
            # 在visdom上显示当前损失
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] *
                           len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    # 打印当前损失
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """打印当前损失到控制台；也将损失保存到磁盘

        Parameters:
            epoch (int) -- 当前epoch
            iters (int) -- 当前epoch的训练迭代次数（每个epoch结束时重置为0）
            losses (OrderedDict) -- 训练损失存储在（名称，浮点数）对的格式中
            t_comp (float) -- 每个数据点（按batch_size归一化）的计算时间
            t_data (float) -- 每个数据点（按batch_size归一化）的数据加载时间
        """
        # 创建消息
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
            epoch, iters, t_comp, t_data)
        # 遍历losses中的每个损失
        for k, v in losses.items():
            # 将损失添加到消息中
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        # 打开日志文件
        with open(self.log_name, "a") as log_file:
            # 写入日志
            log_file.write('%s\n' % message)  # save the message
