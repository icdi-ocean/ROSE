import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import spectral
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count, set_start_method
from scipy.optimize import nnls
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading
import multiprocessing
from matplotlib import rcParams
import logging

# 设置日志记录
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)


rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体，以支持中文显示
rcParams['axes.unicode_minus'] = False

mineral_translation = {
    "Acmite": "霓石", "Actinolite": "透闪石", "Adularia": "正长石", "Albite": "钠长石", "Allanite": "硅铈钙石",
    "Almandine": "铁铝榴石", "Alunite": "明矾石", "Ammonio-Illite-Smect": "铵伊利石-蒙脱石", "Ammonio-Jarosite": "铵黄钾铁矾", "Ammonio-Smectite": "铵蒙脱石",
    "Amphibole": "角闪石", "Analcime": "方沸石", "Ancylite": "水碳铈钙石", "Andalusite": "红柱石", "Andesine": "中长石",
    "Andradite": "钙铁榴石", "Anhydrite": "硬石膏", "Annite": "富铁黑云母", "Anorthite": "钙长石", "Anthophyllite": "直闪石",
    "Antigorite": "蛇纹石", "Aragonite": "文石", "Arsenopyrite": "毒砂", "Augite": "辉石", "Axinite": "钛硼钙铝矽酸盐",
    "Azurite": "蓝铜矿", "Barite": "重晶石", "Bassanite": "水石膏", "Bastnaesite": "氟碳铈矿", "Beidellite": "北山石",
    "Beryl": "绿柱石", "Biotite": "黑云母", "Bloedite": "镁钠硫酸盐", "Bronzite": "铜辉石", "Brookite": "锐钛矿",
    "Brucite": "水镁石", "Buddingtonite": "氨硅石", "Butlerite": "钠硫酸铁矿", "Bytownite": "比霰石", "Calcite": "方解石",
    "Carbon": "碳", "Carnallite": "光卤石", "Cassiterite": "锡石", "Celestite": "天青石", "Chalcedony": "玉髓",
    "Chalcopyrite": "黄铜矿", "Chert": "燧石", "Chlorapatite": "氯磷灰石", "Chlorite": "绿泥石", "Chromite": "铬铁矿",
    "Chrysocolla": "硅孔雀石", "Chrysotile": "温石棉", "Cinnabar": "辰砂", "Clinochlore": "斜绿泥石", "Clinoptilolite": "沸石",
    "Clintonite": "红绿泥石", "Colemanite": "钙硼酸盐", "Cookeite": "锂绿泥石", "Copiapite": "铜酸铁矾", "Coquimbite": "三水合硫酸铁",
    "Corrensite": "堇青石", "Corundum": "刚玉", "Cronstedtite": "镁蛇纹石", "Cummingtonite": "云母蛇纹石", "Cuprite": "红铜矿",
    "Datolite": "硼硅钙石", "Diaspore": "水铝石", "Dickite": "玉桂石", "Diopside": "透辉石", "Dipyre": "双柱石",
    "Dolomite": "白云石", "Dumortierite": "硬铝硅石", "Elbaite": "电气石", "Endellite": "白云石", "Enstatite": "锂云母",
    "Epidote": "绿帘石", "Epsomite": "泻盐", "Eugsterite": "硼钙镁矿", "Fassaite": "熔融钙铝硅酸盐", "Ferrihydrite": "水合氧化铁",
    "Fluorapatite": "氟磷灰石", "Forsterite": "镁橄榄石", "Galena": "方铅矿", "Gaylussite": "碳酸钠钙", "Gibbsite": "氢氧化铝",
    "Glauconite": "绿帘石", "Glaucophane": "蓝闪石", "Goethite": "针铁矿", "Grossular": "钙铝榴石", "Gypsum": "石膏",
    "Halite": "岩盐", "Halloysite": "哈洛石", "Hectorite": "富锂膨润土", "Hedenbergite": "钙铁辉石", "Hematite": "赤铁矿",
    "Heulandite": "方沸石", "Hornblende": "角闪石", "Howlite": "电气石", "Hydrogrossular": "水铬榴石", "Hydroxyl-Apatite": "羟基磷灰石",
    "Hypersthene": "透闪石", "Illite": "伊利石", "Ilmenite": "钛铁矿", "Jadeite": "硬玉", "Jarosite": "黄钾铁矾",
    "Kainite": "钾镁硫酸盐", "Kaolinite": "高岭石", "Karpatite": "卡帕石", "Kieserite": "苦盐", "Knebelite": "铬铁矿",
    "Labradorite": "拉长石", "Laumontite": "方沸石", "Lazurite": "天青石", "Lepidocrocite": "石楠", "Lepidolite": "锂云母",
    "Limonite": "褐铁矿", "Lizardite": "蜥蜴石", "Maghemite": "磁铁矿", "Magnetite": "磁铁矿", "Malachite": "孔雀石",
    "Margarite": "硬柱石", "Marialite": "钠钙云母", "Mascagnite": "二水硫酸铵", "Meionite": "钙长石", "Microcline": "微斜长石",
    "Mirabilite": "泻盐", "Mizzonite": "莫卓克矿", "Monazite": "独居石", "Monticellite": "钙橄榄石", "Montmorillonite": "膨润土",
    "Mordenite": "沸石", "Muscovite": "白云母", "Nacrite": "富锂云母", "Nanohematite": "纳米赤铁矿", "Natrolite": "方钠沸石",
    "Nepheline": "霞石", "Nephrite": "软玉", "Niter": "硝石", "Nontronite": "蒙脱石", "Oligoclase": "奥长石",
    "Olivine": "橄榄石", "Opal": "蛋白石", "Orthoclase": "正长石", "Palygorskite": "纤维", "Paragonite": "镁云母",
    "Parisite": "钡铈铀矿", "Pectolite": "硅酸盐", "Perthite": "钾长石", "Phlogopite": "白云母", "Pigeonite": "钠长石",
    "Pinnoite": "硼钙石", "Pitch": "沥青", "Polyhalite": "多盐石", "Portlandite": "氢氧钙石", "Prehnite": "绿帘石",
    "Prochlorite": "绿泥石", "Psilomelane": "褐锰矿", "Pyrite": "黄铁矿", "Pyrolusite": "软锰矿", "Pyromorphite": "绿铅矿",
    "Pyrophyllite": "叶蜡石", "Pyroxene": "辉石", "Pyroxmangite": "锰辉石", "Pyrrhotite": "磁黄铁矿", "Quartz": "石英",
    "Rectorite": "钠蒙脱石", "Rhodochrosite": "菱锰矿", "Rhodonite": "硅锰石", "Richterite": "碳酸钙镁矿", "Riebeckite": "钠铁闪石",
    "Rivadavite": "硬硅石", "Roscoelite": "钒云母", "Rutile": "金红石", "Sanidine": "条纹长石", "Saponite": "钙蒙脱石", "Sauconite": "锌蒙脱石", "Schwertmannite": "针铁矿", "Scolecite": "链沸石",
    "Sepiolite": "海泡石", "Serpentine": "蛇纹石", "Siderite": "菱铁矿", "Siderophyllite": "富铁黑云母", "Smaragdite": "绿柱石",
    "Sodium": "钠", "Spessartine": "锰铝榴石", "Sphalerite": "闪锌矿", "Sphene": "榍石", "Spodumene": "锂辉石",
    "Staurolite": "十字石", "Stilbite": "丝沸石", "Strontianite": "锶矿", "Sulfur": "硫", "Syngenite": "共生石膏",
    "Talc": "滑石", "Thenardite": "钠石", "Thuringite": "绿泥石", "Tincalconite": "钠硼矾", "Topaz": "黄玉",
    "Tourmaline": "电气石", "Tremolite": "透闪石", "Trona": "天然碳酸钠", "Ulexite": "钙硼石", "Uralite": "乌拉石",
    "Uvarovite": "钙铬榴石", "Vermiculite": "蛭石", "Vesuvianite": "黄榴石", "Witherite": "毒重石", "Wollastonite": "硅灰石",
    "Xenotime": "铋石", "Zircon": "锆石", "Zoisite": "黝帘石", "Zunyite": "氯黄晶"
}


# 定义波谱文件夹路径
spectral_directory = "E:/Google load/usgs_splib07/ASCIIdata/ASCIIdata_splib07b_cvASD/ChapterM_Minerals"

# SRACN 模型相关代码
class SpectralAttention(nn.Module):
    def __init__(self, input_channels, reduction=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, max(1, input_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, input_channels // reduction), input_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, channels, length)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x), y  # 返回加权特征和注意力权重

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class SpectralResidualAttentionConvolutionalNetwork(nn.Module):
    def __init__(self, input_length, num_classes):
        super(SpectralResidualAttentionConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.attention1 = SpectralAttention(64)
        self.residual_block1 = ResidualBlock(64, 64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention2 = SpectralAttention(128)
        self.residual_block2 = ResidualBlock(128, 128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.attention3 = SpectralAttention(256)
        self.residual_block3 = ResidualBlock(256, 256)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_class = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, input_length)
        x = x.unsqueeze(1)  # (batch_size, 1, input_length)

        # Layer 1
        x = self.relu(self.bn1(self.conv1(x)))
        x, attn1 = self.attention1(x)
        x = self.residual_block1(x)
        x = F.max_pool1d(x, kernel_size=2)

        # Layer 2
        x = self.relu(self.bn2(self.conv2(x)))
        x, attn2 = self.attention2(x)
        x = self.residual_block2(x)
        x = F.max_pool1d(x, kernel_size=2)

        # Layer 3
        x = self.relu(self.bn3(self.conv3(x)))
        x, attn3 = self.attention3(x)
        x = self.residual_block3(x)
        x = F.max_pool1d(x, kernel_size=2)

        # Global Pooling and Classification
        x = self.global_avg_pool(x).squeeze(-1)  # (batch_size, 256)
        logits = self.fc_class(x)  # (batch_size, num_classes)
        # 使用 Softmax 得到每个矿物的概率
        probabilities = F.softmax(logits, dim=1)

        return probabilities, [attn1, attn2, attn3]

class MineralDataset(Dataset):
    def __init__(self, spectra, labels, spectral_mean=None, spectral_std=None):
        self.spectra = spectra
        self.labels = labels
        self.spectral_mean = spectral_mean
        self.spectral_std = spectral_std

        # 归一化光谱
        if self.spectral_mean is not None and self.spectral_std is not None:
            spectral_normalized = (self.spectra - self.spectral_mean) / self.spectral_std
        else:
            spectral_mean = self.spectra.mean(axis=0, keepdims=True)  # 按波段计算均值
            spectral_std = self.spectra.std(axis=0, keepdims=True) + 1e-8  # 按波段计算标准差
            spectral_normalized = (self.spectra - spectral_mean) / spectral_std
            self.spectral_mean = spectral_mean
            self.spectral_std = spectral_std

        spectral_normalized = np.clip(spectral_normalized, -5, 5)  # 剪裁极端值
        if np.isnan(spectral_normalized).any() or np.isinf(spectral_normalized).any():
            raise ValueError("归一化后的数据包含 NaN 或 Inf。请检查数据预处理步骤。")
        self.inputs = torch.tensor(spectral_normalized, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# 主应用程序类
# 主应用程序类
class ENVIStyleViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ENVI风格影像展示和矿物分析系统")
        self.root.geometry('1500x800')  # 设置窗口大小

        # 初始化影像数据、当前波段和波谱
        self.image_data = None
        self.current_band = 0
        self.analysis_method = None  # 用户选择的分析方法

        # 创建主框架
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        self.control_frame = tk.Frame(self.main_frame, width=450)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack_propagate(False)  # 防止控件改变 frame 大小

        # 右侧显示区域
        self.display_frame = tk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 添加滚动条以支持更多控件
        self.control_canvas = tk.Canvas(self.control_frame)
        self.scrollbar = tk.Scrollbar(self.control_frame, orient="vertical", command=self.control_canvas.yview)
        self.scrollable_frame = tk.Frame(self.control_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )

        self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 打开影像文件按钮
        tk.Button(self.scrollable_frame, text="打开影像文件", command=self.load_image).pack(pady=5)

        # 拉伸类型选择器和应用按钮
        stretch_group = tk.LabelFrame(self.scrollable_frame, text="图像拉伸")
        stretch_group.pack(pady=5, fill="x")

        stretch_frame = tk.Frame(stretch_group)
        stretch_frame.pack()
        tk.Label(stretch_frame, text="选择拉伸类型:").pack(side=tk.LEFT)
        self.stretch_selector = ttk.Combobox(
            stretch_frame,
            state="readonly",
            values=["无", "线性拉伸", "直方图均衡"]
        )
        self.stretch_selector.current(0)
        self.stretch_selector.pack(side=tk.LEFT)
        # 添加拉伸百分比输入
        tk.Label(stretch_frame, text="拉伸百分比 (%):").pack(side=tk.LEFT)
        self.stretch_percent = tk.DoubleVar(value=2.0)  # 默认 2%
        tk.Entry(stretch_frame, textvariable=self.stretch_percent, width=5).pack(side=tk.LEFT)
        tk.Button(stretch_frame, text="应用", command=self.display_band).pack(side=tk.LEFT)

        # 波段选择器
        band_group = tk.LabelFrame(self.scrollable_frame, text="波段选择")
        band_group.pack(pady=5, fill="x")

        tk.Label(band_group, text="选择可视化波段:").pack()
        self.band_selector_r = ttk.Combobox(band_group, state="readonly")
        self.band_selector_g = ttk.Combobox(band_group, state="readonly")
        self.band_selector_b = ttk.Combobox(band_group, state="readonly")
        self.band_selector_r.pack()
        self.band_selector_g.pack()
        self.band_selector_b.pack()

        self.band_selector_r.bind("<<ComboboxSelected>>", self.display_band)
        self.band_selector_g.bind("<<ComboboxSelected>>", self.display_band)
        self.band_selector_b.bind("<<ComboboxSelected>>", self.display_band)
        self.label_colors = {}  # 标签名称到颜色的映射
        self.color_index = 0  # 用于分配颜色的索引
        self.colors_list = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta']  # 可扩展的颜色列表

        # 矿物选择方式选项
        mineral_group = tk.LabelFrame(self.scrollable_frame, text="矿物选择")
        mineral_group.pack(pady=5, fill="x")

        tk.Label(mineral_group, text="选择矿物输入方式:").pack()
        self.mineral_input_method = tk.StringVar(value="列表选择")
        tk.Radiobutton(mineral_group, text="列表选择", variable=self.mineral_input_method, value="列表选择", command=self.switch_mineral_input_method).pack()
        tk.Radiobutton(mineral_group, text="文本输入", variable=self.mineral_input_method, value="文本输入", command=self.switch_mineral_input_method).pack()

        # 矿物选择控件（列表选择）
        self.mineral_list_frame = tk.Frame(mineral_group)
        self.mineral_list_frame.pack()
        tk.Label(self.mineral_list_frame, text="选择矿物（可多选）:").pack()
        # 创建带搜索功能的列表框
        search_frame = tk.Frame(self.mineral_list_frame)
        search_frame.pack()
        tk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        self.mineral_search_var = tk.StringVar()
        tk.Entry(search_frame, textvariable=self.mineral_search_var).pack(side=tk.LEFT)
        self.mineral_search_var.trace("w", self.update_mineral_list)

        self.mineral_listbox = tk.Listbox(self.mineral_list_frame, selectmode=tk.MULTIPLE, exportselection=0, width=40)
        self.mineral_listbox.pack()

        # 获取矿物列表并显示
        self.mineral_names = self.get_unique_mineral_names()
        self.update_mineral_list()

        # 矿物选择控件（文本输入）
        self.mineral_entry_frame = tk.Frame(mineral_group)
        # 默认隐藏文本输入框
        self.mineral_entry_frame.pack_forget()
        tk.Label(self.mineral_entry_frame, text="输入矿物名称（可输入多个，用逗号分隔）:").pack()
        self.mineral_entry = tk.Entry(self.mineral_entry_frame)
        self.mineral_entry.pack()

        # 分析方法选择器
        analysis_group = tk.LabelFrame(self.scrollable_frame, text="分析设置")
        analysis_group.pack(pady=5, fill="x")

        tk.Label(analysis_group, text="选择分析方法:").pack()
        self.analysis_selector = ttk.Combobox(
            analysis_group,
            state="readonly",
            values=[
                "光谱角匹配 (SAM)",
                "端元提取 (N-FINDR)",
                "端元提取 (PPI)",
                "端元提取 (ATGP)",
                "线性回归解混",
                "非负最小二乘法 (FCLS)",
                "主成分分析 (PCA)",
                "SRACN 模型预测"
            ],
        )
        self.analysis_selector.pack()
        self.analysis_selector.bind("<<ComboboxSelected>>", self.update_parameters)

        # 参数设置区域
        self.param_frame = tk.Frame(analysis_group)
        self.param_frame.pack(pady=5)

        # 运行分析按钮
        self.run_analysis_button = tk.Button(analysis_group, text="运行分析", command=self.run_analysis)
        self.run_analysis_button.pack()

        # 绘制波谱按钮
        tk.Button(analysis_group, text="绘制矿物波谱", command=self.plot_spectrum).pack()

        # 自动矿物识别按钮
        tk.Button(analysis_group, text="自动矿物识别", command=self.run_automatic_recognition).pack(pady=5)

        # 添加标注模式选择
        tk.Label(self.scrollable_frame, text="标注模式:").pack(pady=5)
        self.label_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(self.scrollable_frame, text="启用标注模式", variable=self.label_mode,
                       command=self.toggle_label_mode).pack()
        tk.Button(self.scrollable_frame, text="确定标注", command=self.confirm_labels).pack(pady=5)

        # 保存结果按钮
        tk.Button(self.scrollable_frame, text="保存结果", command=self.save_result).pack(pady=5)

        # 初始化参数变量
        self.threshold = tk.DoubleVar(value=0.1)
        self.num_endmembers = tk.IntVar(value=5)
        self.visualization_threshold = tk.DoubleVar(value=0.8)  # 可视化阈值
        self.num_epochs = tk.IntVar(value=10)  # 训练轮数

        # 保存结果的变量
        self.result_image = None  # 用于保存结果图像
        self.result_data = None  # 用于保存结果数据

        # 地理信息变量
        self.geo_info = None

        # 像元光谱窗口
        self.spectrum_window = None
        self.spectra_plotted = {}  # {(x, y): line}

        # 标注数据
        self.labels = {}  # {(x, y): label}
        self.selected_pixels = []  # 存储多选的像素坐标

        # 进度条窗口
        self.progress_window = None
        self.progress_var = None

        # 添加滚轮绑定到左侧控制面板
        self.control_canvas.bind_all("<MouseWheel>", self.on_control_scroll)

    def on_control_scroll(self, event):
        # 滚轮向前（向上）
        if event.delta > 0:
            self.control_canvas.yview_scroll(-1, "units")
        # 滚轮向后（向下）
        else:
            self.control_canvas.yview_scroll(1, "units")

    def extract_mineral_name_from_filename(self, filename):
        # 假设文件名格式为：s07_ASD_Actinolite_HS116.3B_BECKb_AREF.txt
        # 我们提取第三个下划线分隔的部分作为矿物名称
        base_name = os.path.basename(filename)
        parts = base_name.split('_')
        if len(parts) >= 3:
            mineral_name = parts[2]
            return mineral_name
        else:
            return None

    def get_unique_mineral_names(self):
        mineral_names_set = set()
        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for root_dir, dirs, files in os.walk(spectral_directory):
                    for filename in files:
                        if filename.endswith('.txt'):
                            futures.append(executor.submit(self.extract_mineral_name_from_filename, filename))
                for future in futures:
                    mineral_name = future.result()
                    if mineral_name:
                        # 支持中文名称
                        for eng_name, chi_name in mineral_translation.items():
                            if mineral_name.lower() == eng_name.lower():
                                mineral_name = chi_name
                                break
                        mineral_names_set.add(mineral_name)
        except Exception as e:
            logging.error("获取矿物名称失败", exc_info=True)
            messagebox.showerror("错误", f"获取矿物名称失败: {e}")
        return sorted(mineral_names_set)

    def switch_mineral_input_method(self):
        method = self.mineral_input_method.get()
        if method == "列表选择":
            self.mineral_entry_frame.pack_forget()
            self.mineral_list_frame.pack()
        else:
            self.mineral_list_frame.pack_forget()
            self.mineral_entry_frame.pack()

    def update_mineral_list(self, *args):
        search_term = self.mineral_search_var.get().lower()
        # 获取当前选择的矿物名称集合
        current_selection = set()
        for idx in self.mineral_listbox.curselection():
            mineral_name = self.mineral_listbox.get(idx)
            current_selection.add(mineral_name)
        self.mineral_listbox.delete(0, tk.END)
        for mineral in self.mineral_names:
            if search_term in mineral.lower():
                self.mineral_listbox.insert(tk.END, mineral)
                if mineral in current_selection:
                    idx = self.mineral_listbox.size() - 1  # 获取最后插入项的索引
                    self.mineral_listbox.selection_set(idx)

    def load_image(self):
        filepath = filedialog.askopenfilename(title="选择影像文件", filetypes=[("ENVI files", "*.dat"), ("ENVI headers", "*.hdr")])
        if not filepath:
            return

        try:
            if filepath.endswith('.dat'):
                hdr_filepath = filepath.replace('.dat', '.hdr')
            elif filepath.endswith('.hdr'):
                hdr_filepath = filepath
                filepath = filepath.replace('.hdr', '.dat')
            else:
                messagebox.showerror("错误", "请选择有效的 ENVI 格式文件（.dat 或 .hdr）。")
                return

            if not os.path.exists(hdr_filepath):
                messagebox.showerror("错误", "找不到对应的 .hdr 文件。")
                return

            img = spectral.envi.open(hdr_filepath, filepath)
            self.image_data = np.array(img.load())

            self.geo_info = self.parse_hdr_file(hdr_filepath)

            bands = [f"Band {i + 1}" for i in range(self.image_data.shape[2])]
            self.band_selector_r["values"] = bands
            self.band_selector_g["values"] = bands
            self.band_selector_b["values"] = bands
            self.band_selector_r.current(0)
            self.band_selector_g.current(1)
            self.band_selector_b.current(2)

            self.current_band = 0

            self.display_band()
            messagebox.showinfo("加载成功", "影像数据已成功加载！")

        except Exception as e:
            logging.error("加载影像数据失败", exc_info=True)
            messagebox.showerror("错误", f"加载影像数据失败: {e}")

    def parse_hdr_file(self, hdr_filepath):
        geo_info = {}
        try:
            with open(hdr_filepath, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        geo_info[key] = value
        except Exception as e:
            logging.error("解析HDR文件失败", exc_info=True)
            messagebox.showerror("错误", f"解析HDR文件失败: {e}")
        return geo_info

    def get_geo_transform(self):
        map_info = self.geo_info.get('map info', None)
        if map_info:
            parts = map_info.strip('{}').split(',')
            x_start = float(parts[3])
            y_start = float(parts[4])
            x_size = float(parts[5])
            y_size = float(parts[6])
            transform = from_origin(x_start, y_start, x_size, -y_size)
            crs = CRS.from_string('EPSG:4326')
            return transform, crs
        else:
            transform = from_origin(0, 0, 1, 1)
            crs = CRS.from_string('EPSG:4326')
            return transform, crs

    def get_image_extent(self):
        map_info = self.geo_info.get('map info', None)
        if map_info:
            parts = map_info.strip('{}').split(',')
            x_start = float(parts[3])
            y_start = float(parts[4])
            x_size = float(parts[5])
            y_size = float(parts[6])
            x_end = x_start + x_size * self.image_data.shape[1]
            y_end = y_start - y_size * self.image_data.shape[0]
            return [x_start, x_end, y_end, y_start]
        else:
            return None

    def apply_stretch(self, data):
        stretch_type = self.stretch_selector.get()
        percent = self.stretch_percent.get()
        if stretch_type == "无":
            return data
        elif stretch_type == "线性拉伸":
            lower_bound = np.percentile(data, percent)
            upper_bound = np.percentile(data, 100 - percent)
            stretched_data = np.clip(data, lower_bound, upper_bound)
            stretched_data = (stretched_data - lower_bound) / (upper_bound - lower_bound + 1e-6) * 255
            return stretched_data
        elif stretch_type == "直方图均衡":
            data_flat = data.flatten()
            hist, bins = np.histogram(data_flat, bins=256, range=[np.min(data_flat), np.max(data_flat)])
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            data_equalized = np.interp(data_flat, bins[:-1], cdf_normalized)
            return data_equalized.reshape(data.shape)
        else:
            return data

    def display_band(self, event=None):
        if self.image_data is None:
            return

        try:
            for widget in self.display_frame.winfo_children():
                widget.destroy()

            band_r = self.band_selector_r.current()
            band_g = self.band_selector_g.current()
            band_b = self.band_selector_b.current()

            img_r = self.image_data[:, :, band_r]
            img_g = self.image_data[:, :, band_g]
            img_b = self.image_data[:, :, band_b]

            rgb_image = np.dstack((img_r, img_g, img_b))

            rgb_image = self.apply_stretch(rgb_image)
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

            self.display_image(rgb_image)

            self.result_image = Image.fromarray(rgb_image)
            self.result_data = rgb_image

        except Exception as e:
            logging.error("显示图像失败", exc_info=True)
            messagebox.showerror("错误", f"显示图像失败: {e}")

    def display_image(self, rgb_image):
        self.zoom_level = 1.0

        image = Image.fromarray(rgb_image)
        self.original_image = image.copy()

        self.canvas = tk.Canvas(self.display_frame, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.tk_image = ImageTk.PhotoImage(image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # 绑定鼠标滚轮事件
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def zoom(self, event):
        # 计算缩放因子
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1

        # 限制缩放级别
        self.zoom_level = max(0.1, min(self.zoom_level, 10))

        # 获取鼠标位置
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # 缩放图像
        width, height = self.original_image.size
        new_size = (int(width * self.zoom_level), int(height * self.zoom_level))
        resized_image = self.original_image.resize(new_size, Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

        # 调整 Canvas 大小
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_click(self, event):
        if self.image_data is None:
            return

        x = int(self.canvas.canvasx(event.x) / self.zoom_level)
        y = int(self.canvas.canvasy(event.y) / self.zoom_level)

        if x < 0 or y < 0 or x >= self.image_data.shape[1] or y >= self.image_data.shape[0]:
            return

        if self.label_mode.get():
            # 标注模式，开始记录拖动区域
            self.start_x = x
            self.start_y = y
            self.rect = self.canvas.create_rectangle(
                x * self.zoom_level, y * self.zoom_level,
                x * self.zoom_level, y * self.zoom_level,
                outline=self.get_label_color()
            )
        else:
            # 显示光谱
            pixel_spectrum = self.image_data[y, x, :]

            if self.spectrum_window is None or not tk.Toplevel.winfo_exists(self.spectrum_window):
                self.spectrum_window = tk.Toplevel(self.root)
                self.spectrum_window.title("像元光谱")
                self.fig, self.ax = plt.subplots(figsize=(8, 6))
                self.canvas_spectrum = FigureCanvasTkAgg(self.fig, master=self.spectrum_window)
                self.canvas_spectrum.draw()
                self.canvas_spectrum.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # 添加保存波谱按钮
                save_button = tk.Button(self.spectrum_window, text="保存波谱为TXT", command=self.save_spectrum)
                save_button.pack(pady=5)

                self.spectrum_data = {}
            else:
                self.ax = self.fig.axes[0]
                self.ax.clear()

            # 归一化反射率
            reflectance = (pixel_spectrum - np.min(pixel_spectrum)) / (
                        np.max(pixel_spectrum) - np.min(pixel_spectrum) + 1e-6)
            # 生成一个颜色
            color = self.get_next_color()

            # 波长从350开始，每个点增加1 nm
            wavelengths = np.arange(350, 350 + len(pixel_spectrum))

            self.ax.plot(wavelengths, reflectance, label=f"像素 ({x}, {y})", color=color)
            self.ax.set_xlabel("波长 (nm)")
            self.ax.set_ylabel("反射率")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas_spectrum.draw()

            # 保存波谱数据
            self.spectrum_data[f"({x}, {y})"] = (wavelengths, reflectance, color)

    def on_drag(self, event):
        if self.label_mode.get() and hasattr(self, 'rect'):
            x = int(self.canvas.canvasx(event.x) / self.zoom_level)
            y = int(self.canvas.canvasy(event.y) / self.zoom_level)
            self.canvas.coords(
                self.rect,
                self.start_x * self.zoom_level, self.start_y * self.zoom_level,
                x * self.zoom_level, y * self.zoom_level
            )

    def on_release(self, event):
        if self.label_mode.get() and hasattr(self, 'rect'):
            x_end = int(self.canvas.canvasx(event.x) / self.zoom_level)
            y_end = int(self.canvas.canvasy(event.y) / self.zoom_level)

            x1, x2 = sorted([self.start_x, x_end])
            y1, y2 = sorted([self.start_y, y_end])

            # 将选定区域的像素坐标添加到 selected_pixels 列表
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    if (x, y) not in self.selected_pixels:
                        self.selected_pixels.append((x, y))

    def confirm_labels(self):
        if not self.selected_pixels:
            messagebox.showwarning("警告", "没有选定任何像元进行标注。")
            return

        label = simpledialog.askstring("标注", f"为选定的像素输入标签：")
        if label:
            for x, y in self.selected_pixels:
                self.labels[(x, y)] = label
                # 在图像上标记
                self.canvas.create_rectangle(
                    x * self.zoom_level, y * self.zoom_level,
                    (x + 1) * self.zoom_level, (y + 1) * self.zoom_level,
                    outline="red"
                )
            self.selected_pixels = []
            messagebox.showinfo("标注完成", "像素标注已完成。")
        else:
            messagebox.showwarning("警告", "标注标签不能为空。")

    def toggle_label_mode(self):
        if self.label_mode.get():
            messagebox.showinfo("标注模式", "已开启标注模式。点击并拖动以选择像素区域。")
        else:
            messagebox.showinfo("标注模式", "已关闭标注模式。")
            self.selected_pixels = []

    def get_selected_minerals(self):
        method = self.mineral_input_method.get()
        if method == "列表选择":
            selected_indices = self.mineral_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("错误", "请先选择矿物。")
                return None
            selected_minerals = []
            for index in selected_indices:
                mineral_name = self.mineral_listbox.get(index)
                selected_minerals.append(mineral_name)
            return selected_minerals
        else:
            mineral_input = self.mineral_entry.get()
            if not mineral_input.strip():
                messagebox.showerror("错误", "请先输入矿物名称。")
                return None
            # 支持中文逗号和英文逗号
            mineral_names = [name.strip() for name in re.split('[,，]', mineral_input)]
            return mineral_names

    def plot_spectrum(self):
        mineral_names = self.get_selected_minerals()
        if mineral_names is None:
            return

        for widget in self.display_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 6))

        for mineral_name in mineral_names:
            spectra_list = self.get_mineral_spectra_by_name(mineral_name)
            if not spectra_list:
                continue

            x_values = np.arange(350, 350 + spectra_list[0].shape[0])

            for i, spectrum in enumerate(spectra_list):
                # 归一化反射率
                reflectance = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-6)
                color = self.get_next_color()
                label = f"{mineral_name} 光谱 {i + 1}"  # 使用矿物名称和光谱索引作为标签
                ax.plot(x_values, reflectance, label=label, color=color)

        ax.set_xlabel("波长 (nm)")
        ax.set_ylabel("反射率")
        ax.set_title("矿物光谱反射率")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_parameters(self, event=None):
        method = self.analysis_selector.get()

        for widget in self.param_frame.winfo_children():
            widget.destroy()

        if method in ["光谱角匹配 (SAM)", "非负最小二乘法 (FCLS)", "线性回归解混"]:
            tk.Label(self.param_frame, text="阈值 (SAM 阈值 / 丰度阈值):").pack(side=tk.LEFT)
            tk.Entry(self.param_frame, textvariable=self.threshold, width=5).pack(side=tk.LEFT)
            tk.Label(self.param_frame, text="可视化阈值:").pack(side=tk.LEFT)
            tk.Entry(self.param_frame, textvariable=self.visualization_threshold, width=5).pack(side=tk.LEFT)
        elif "端元提取" in method:
            tk.Label(self.param_frame, text="端元数量:").pack(side=tk.LEFT)
            tk.Entry(self.param_frame, textvariable=self.num_endmembers, width=5).pack(side=tk.LEFT)
        elif method == "SRACN 模型预测":
            tk.Label(self.param_frame, text="训练轮数:").pack(side=tk.LEFT)
            self.num_epochs = tk.IntVar(value=10)
            tk.Entry(self.param_frame, textvariable=self.num_epochs, width=5).pack(side=tk.LEFT)
        else:
            pass

    def run_analysis(self):
        if self.image_data is None:
            messagebox.showerror("错误", "请先加载影像数据。")
            return

        method = self.analysis_selector.get()
        analysis_methods = {
            "光谱角匹配 (SAM)": self.run_sam,
            "端元提取 (N-FINDR)": self.run_n_findr,
            "端元提取 (PPI)": self.run_ppi,
            "端元提取 (ATGP)": self.run_atgp,
            "线性回归解混": self.run_linear_unmixing,
            "非负最小二乘法 (FCLS)": self.run_fcls,
            "主成分分析 (PCA)": self.run_pca,
            "SRACN 模型预测": self.run_sracn_model
        }

        if method in analysis_methods:
            # 禁用分析按钮以防重复点击
            self.run_analysis_button.config(state=tk.DISABLED)
            thread = threading.Thread(target=analysis_methods[method])
            thread.start()
        else:
            messagebox.showerror("错误", "请选择一个有效的分析方法。")

    def run_sam(self):
        mineral_names = self.get_selected_minerals()
        if mineral_names is None:
            self.run_analysis_button.config(state=tk.NORMAL)
            return

        threshold = self.threshold.get()
        visualization_threshold = self.visualization_threshold.get()

        try:
            for widget in self.display_frame.winfo_children():
                widget.destroy()

            final_map = np.zeros((self.image_data.shape[0], self.image_data.shape[1]), dtype=int)
            num_minerals = len(mineral_names)

            # 预处理影像数据
            image_data = self.image_data.reshape(-1, self.image_data.shape[2])
            image_norm = np.linalg.norm(image_data, axis=1, keepdims=True)
            image_norm[image_norm == 0] = 1e-6

            for idx, mineral_name in enumerate(mineral_names):
                spectra_list = self.get_mineral_spectra_by_name(mineral_name)
                if not spectra_list:
                    continue

                # 将所有矿物光谱堆叠成一个矩阵
                spectra_matrix = np.vstack(spectra_list)
                spectra_norm = np.linalg.norm(spectra_matrix, axis=1, keepdims=True)
                spectra_norm[spectra_norm == 0] = 1e-6

                # 计算余弦相似度
                cos_theta = np.dot(image_data, spectra_matrix.T) / (image_norm * spectra_norm.T)
                cos_theta = np.clip(cos_theta, -1, 1)
                sam_angles = np.arccos(cos_theta)
                sam_map = np.min(sam_angles, axis=1).reshape(self.image_data.shape[0], self.image_data.shape[1])

                classification_map = (sam_map < threshold).astype(int)
                display_map = (classification_map >= visualization_threshold).astype(int) * (idx + 1)
                final_map += display_map

            # 创建RGBA图像
            final_map_rgba = np.zeros((final_map.shape[0], final_map.shape[1], 4), dtype=np.uint8)

            # 定义颜色映射
            color_map = plt.cm.get_cmap('tab20', num_minerals + 1)

            # 将分类结果映射到颜色
            for idx in range(1, num_minerals + 1):
                mask = (final_map == idx)
                color = color_map(idx)
                final_map_rgba[mask] = [int(c * 255) for c in color[:3]] + [255]  # 设置Alpha为255

            # 低于阈值的像素保持透明
            final_map_rgba[final_map == 0, 3] = 0  # Alpha 通道为 0，表示完全透明

            # 将原始影像转换为 RGB
            rgb_image = self.image_data[:, :, :3]
            rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image) + 1e-6) * 255
            rgb_image = rgb_image.astype(np.uint8)
            background_image = Image.fromarray(rgb_image)

            # 创建叠加图像
            overlay_image = Image.fromarray(final_map_rgba, mode='RGBA')

            # 将叠加图像与背景图像合并
            result_image = Image.alpha_composite(background_image.convert('RGBA'), overlay_image)

            # 获取地理范围
            extent = self.get_image_extent()

            # 显示结果
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(result_image, extent=extent)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title("SAM 分类结果")
            ax.axis('on')  # 显示坐标轴

            # 创建颜色条
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=color_map), ax=ax,
                                ticks=np.linspace(0.5, num_minerals + 0.5, num_minerals + 1))
            cbar_labels = ['未分类'] + [name for name in mineral_names]
            cbar.ax.set_yticklabels(cbar_labels)

            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.result_data = np.array(result_image)
            self.result_image = result_image

        except Exception as e:
            logging.error("SAM 分类失败", exc_info=True)
            messagebox.showerror("错误", f"SAM 分类失败: {e}")
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def run_n_findr(self):
        num_endmembers = self.num_endmembers.get()
        try:
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
            num_pixels = data_reshaped.shape[0]
            bands = data_reshaped.shape[1]

            np.random.seed(0)
            endmember_indices = np.random.choice(num_pixels, num_endmembers, replace=False)
            endmembers = data_reshaped[endmember_indices, :]

            max_volume = 0
            for iteration in range(3):
                for i in range(num_endmembers):
                    for j in range(num_pixels):
                        test_endmembers = np.copy(endmembers)
                        test_endmembers[i, :] = data_reshaped[j, :]
                        volume = self.calculate_simplex_volume(test_endmembers)
                        if volume > max_volume:
                            max_volume = volume
                            endmembers[i, :] = data_reshaped[j, :]
            self.endmembers = endmembers
            self.display_endmembers(endmembers, "N-FINDR 端元提取")
        except Exception as e:
            logging.error("端元提取失败", exc_info=True)
            messagebox.showerror("错误", f"端元提取失败: {e}")
        finally:
            self.run_analysis_button.config(state=tk.NORMAL)

    def calculate_simplex_volume(self, endmembers):
        matrix = endmembers[1:] - endmembers[0]
        try:
            volume = np.abs(np.linalg.det(matrix @ matrix.T)) ** 0.5
        except np.linalg.LinAlgError:
            volume = 0
        return volume

    def run_ppi(self):
        num_endmembers = self.num_endmembers.get()
        num_skewers = 10000
        try:
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
            num_pixels = data_reshaped.shape[0]
            bands = data_reshaped.shape[1]

            mean_vector = np.mean(data_reshaped, axis=0)
            data_centered = data_reshaped - mean_vector

            counts = np.zeros(num_pixels)

            np.random.seed(0)
            for _ in range(num_skewers):
                random_vector = np.random.randn(bands)
                random_vector /= np.linalg.norm(random_vector)

                projections = data_centered @ random_vector

                max_idx = np.argmax(projections)
                min_idx = np.argmin(projections)

                counts[max_idx] += 1
                counts[min_idx] += 1

            endmember_indices = np.argsort(counts)[-num_endmembers:]
            endmembers = data_reshaped[endmember_indices, :]

            self.endmembers = endmembers
            self.display_endmembers(endmembers, "PPI 端元提取")
        except Exception as e:
            logging.error("端元提取失败", exc_info=True)
            messagebox.showerror("错误", f"端元提取失败: {e}")
        finally:
            self.run_analysis_button.config(state=tk.NORMAL)

    def run_atgp(self):
        num_endmembers = self.num_endmembers.get()
        try:
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
            num_pixels = data_reshaped.shape[0]
            bands = data_reshaped.shape[1]

            endmembers = []

            norms = np.linalg.norm(data_reshaped, axis=1)
            first_idx = np.argmax(norms)
            endmembers.append(data_reshaped[first_idx])

            for _ in range(1, num_endmembers):
                U = np.array(endmembers).T
                P_U = U @ np.linalg.pinv(U.T @ U) @ U.T
                residuals = data_reshaped - (P_U @ data_reshaped.T).T
                residual_norms = np.linalg.norm(residuals, axis=1)

                new_idx = np.argmax(residual_norms)
                endmembers.append(data_reshaped[new_idx])

            endmembers = np.array(endmembers)
            self.endmembers = endmembers
            self.display_endmembers(endmembers, "ATGP 端元提取")
        except Exception as e:
            logging.error("端元提取失败", exc_info=True)
            messagebox.showerror("错误", f"端元提取失败: {e}")
        finally:
            self.run_analysis_button.config(state=tk.NORMAL)

    def run_linear_unmixing(self):
        if not hasattr(self, 'endmembers'):
            messagebox.showerror("错误", "请先运行端元提取方法。")
            self.run_analysis_button.config(state=tk.NORMAL)
            return

        endmembers = self.endmembers.T
        try:
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2]).T
            abundances = np.linalg.lstsq(endmembers, data_reshaped, rcond=None)[0]
            abundance_maps = abundances.T.reshape(self.image_data.shape[0], self.image_data.shape[1], -1)
            self.display_abundance_map(abundance_maps, "线性回归解混")
        except Exception as e:
            logging.error("解混失败", exc_info=True)
            messagebox.showerror("错误", f"解混失败: {e}")
        finally:
            self.run_analysis_button.config(state=tk.NORMAL)

    def run_fcls(self):
        if not hasattr(self, 'endmembers'):
            messagebox.showerror("错误", "请先运行端元提取方法。")
            self.run_analysis_button.config(state=tk.NORMAL)
            return

        endmembers = self.endmembers
        try:
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
            num_pixels = data_reshaped.shape[0]
            num_endmembers = endmembers.shape[0]

            abundances = np.zeros((num_pixels, num_endmembers))

            num_cores = cpu_count()
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                chunk_size = int(num_pixels / num_cores)
                chunks = np.array_split(data_reshaped, num_cores)
                futures = [executor.submit(self.nnls_solver_batch, endmembers.T, chunk) for chunk in chunks]
                results = [future.result() for future in futures]
                abundances = np.vstack(results)

            abundance_maps = abundances.reshape(self.image_data.shape[0], self.image_data.shape[1], -1)
            self.display_abundance_map(abundance_maps, "FCLS 解混")
        except Exception as e:
            logging.error("解混失败", exc_info=True)
            messagebox.showerror("错误", f"解混失败: {e}")
        finally:
            self.run_analysis_button.config(state=tk.NORMAL)

    def nnls_solver_batch(self, endmembers_T, pixels_batch):
        abundances_batch = np.zeros((pixels_batch.shape[0], endmembers_T.shape[1]))
        for i, pixel in enumerate(pixels_batch):
            abundance, _ = nnls(endmembers_T, pixel)
            abundances_batch[i] = abundance
        return abundances_batch

    def run_pca(self):
        data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
        try:
            mean_vector = np.mean(data_reshaped, axis=0)
            data_centered = data_reshaped - mean_vector

            covariance_matrix = np.cov(data_centered, rowvar=False)

            eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

            sorted_index = np.argsort(eigen_values)[::-1]
            sorted_eigenvectors = eigen_vectors[:, sorted_index]

            n_components = 3
            eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

            pca_result = np.dot(data_centered, eigenvector_subset)

            pca_image = pca_result.reshape(self.image_data.shape[0], self.image_data.shape[1], n_components)
            self.display_pca_image(pca_image)
        except Exception as e:
            logging.error("PCA 分析失败", exc_info=True)
            messagebox.showerror("错误", f"PCA 分析失败: {e}")
        finally:
            self.run_analysis_button.config(state=tk.NORMAL)

    def get_mineral_spectra_by_name(self, mineral_name):
        if not mineral_name:
            return []

        # 通过中文名称获取对应的英文名称
        eng_name = None
        for key, value in mineral_translation.items():
            if value == mineral_name:
                eng_name = key
                break
        if eng_name is None:
            eng_name = mineral_name  # 如果没有对应，使用原名

        spectra_list = []
        try:
            target_files = []
            for root_dir, dirs, files in os.walk(spectral_directory):
                for filename in files:
                    if filename.endswith('.txt'):
                        extracted_name = self.extract_mineral_name_from_filename(filename)
                        if extracted_name and extracted_name.lower() == eng_name.lower():
                            target_file = os.path.join(root_dir, filename)
                            target_files.append(target_file)

            def load_spectrum(file):
                try:
                    mineral_spectrum = np.loadtxt(file, skiprows=1)
                    mineral_spectrum[mineral_spectrum < 0] = 0
                    if mineral_spectrum.shape[0] != self.image_data.shape[2]:
                        mineral_spectrum = np.interp(
                            np.linspace(0, 1, self.image_data.shape[2]),
                            np.linspace(0, 1, mineral_spectrum.shape[0]),
                            mineral_spectrum
                        )
                    return mineral_spectrum
                except Exception as e:
                    logging.error(f"加载光谱文件失败: {file}", exc_info=True)
                    return None

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(load_spectrum, target_files))

            spectra_list = [s for s in results if s is not None]

            if not spectra_list:
                messagebox.showerror("错误", f"未找到矿物 '{mineral_name}' 的波谱文件。")
        except Exception as e:
            logging.error("获取矿物波谱失败", exc_info=True)
            messagebox.showerror("错误", f"获取矿物波谱失败: {e}")
        return spectra_list

    def display_endmembers(self, endmembers, title):
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, spectrum in enumerate(endmembers):
            x_values = np.arange(350, 350 + len(spectrum))
            ax.plot(x_values, spectrum, label=f'端元 {i + 1}')
        ax.set_xlabel("波长 (nm)")
        ax.set_ylabel("反射率")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.result_data = endmembers
        self.result_image = None

    def display_abundance_map(self, abundance_maps, method_name):
        visualization_threshold = self.visualization_threshold.get()

        for widget in self.display_frame.winfo_children():
            widget.destroy()

        # 将多个端元的丰度图合并为一张图
        num_endmembers = abundance_maps.shape[2]
        final_map = np.zeros((self.image_data.shape[0], self.image_data.shape[1]), dtype=int)
        for idx in range(num_endmembers):
            abundance_map = abundance_maps[:, :, idx]
            mask = abundance_map >= visualization_threshold
            final_map[mask] = idx + 1

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap('tab20', num_endmembers + 1)

        cax = ax.imshow(final_map, cmap=cmap, vmin=0, vmax=num_endmembers)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        cbar = fig.colorbar(cax, ax=ax, ticks=range(num_endmembers + 1))
        cbar_labels = ['低于阈值'] + [f'端元 {i+1}' for i in range(num_endmembers)]
        cbar.ax.set_yticklabels(cbar_labels)
        ax.set_title(f"{method_name} 结果 (可视化阈值={visualization_threshold})")

        canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.result_data = final_map
        self.result_image = None

    def display_pca_image(self, pca_image):
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        self.zoom_level = 1.0

        norm_pca_image = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image) + 1e-6)
        rgb_image = (norm_pca_image[:, :, :3] * 255).astype(np.uint8)

        image = Image.fromarray(rgb_image)
        self.original_image = image.copy()

        self.canvas = tk.Canvas(self.display_frame, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.tk_image = ImageTk.PhotoImage(image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # 绑定鼠标滚轮事件
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.result_image = image
        self.result_data = rgb_image

    def run_automatic_recognition(self):
        try:
            sam_threshold = simpledialog.askfloat("输入", "请输入 SAM 分类阈值（弧度）：", minvalue=0.0)
            if sam_threshold is None:
                return
            visualization_threshold = simpledialog.askfloat("输入", "请输入可视化阈值：", minvalue=0.0)
            if visualization_threshold is None:
                return
            num_minerals = simpledialog.askinteger("输入", "请输入要显示的矿物个数：", minvalue=1)
            if num_minerals is None:
                return

            # 禁用分析按钮以防重复点击
            self.run_analysis_button.config(state=tk.DISABLED)

            # 启动后台线程进行自动识别
            thread = threading.Thread(target=self.automatic_recognition_task, args=(sam_threshold, visualization_threshold, num_minerals))
            thread.start()
        except Exception as e:
            logging.error("自动矿物识别失败", exc_info=True)
            messagebox.showerror("错误", f"自动矿物识别失败: {e}")
            self.run_analysis_button.config(state=tk.NORMAL)

    def automatic_recognition_task(self, sam_threshold, visualization_threshold, num_minerals):
        try:
            mineral_counts = {}
            classification_maps = {}
            for mineral_name in self.mineral_names:
                spectra_list = self.get_mineral_spectra_by_name(mineral_name)
                if not spectra_list:
                    continue

                # 将所有矿物光谱堆叠成一个矩阵
                spectra_matrix = np.vstack(spectra_list)
                spectra_norm = np.linalg.norm(spectra_matrix, axis=1, keepdims=True)
                spectra_norm[spectra_norm == 0] = 1e-6

                # 预处理影像数据
                image_data = self.image_data.reshape(-1, self.image_data.shape[2])
                image_norm = np.linalg.norm(image_data, axis=1, keepdims=True)
                image_norm[image_norm == 0] = 1e-6

                # 计算余弦相似度
                cos_theta = np.dot(image_data, spectra_matrix.T) / (image_norm * spectra_norm.T)
                cos_theta = np.clip(cos_theta, -1, 1)
                sam_angles = np.arccos(cos_theta)
                sam_map = np.min(sam_angles, axis=1).reshape(self.image_data.shape[0], self.image_data.shape[1])

                classification_map = (sam_map < sam_threshold).astype(int)
                count = np.sum(classification_map)
                if count > 0:
                    mineral_counts[mineral_name] = count
                    classification_maps[mineral_name] = classification_map

            sorted_minerals = sorted(mineral_counts.items(), key=lambda x: x[1], reverse=True)
            top_minerals = sorted_minerals[:num_minerals]

            final_map = np.zeros((self.image_data.shape[0], self.image_data.shape[1]), dtype=int)
            for idx, (mineral_name, _) in enumerate(top_minerals):
                classification_map = classification_maps[mineral_name]
                display_map = (classification_map >= visualization_threshold).astype(int) * (idx + 1)
                final_map += display_map

            color_map = plt.get_cmap('tab20', num_minerals + 1)
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(final_map, cmap=color_map, vmin=0, vmax=num_minerals)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            cbar = fig.colorbar(cax, ax=ax, ticks=range(num_minerals + 1))
            cbar_labels = ['未分类'] + [name for name, _ in top_minerals]
            cbar.ax.set_yticklabels(cbar_labels)
            ax.set_title("自动矿物识别结果")

            for widget in self.display_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.result_data = final_map
            self.result_image = None

        except Exception as e:
            logging.error("自动矿物识别失败", exc_info=True)
            messagebox.showerror("错误", f"自动矿物识别失败: {e}")
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def save_result(self):
        if self.result_data is None and self.result_image is None:
            messagebox.showerror("错误", "没有可保存的结果。")
            return

        filetypes = [
            ("GeoTIFF files", "*.tif"),
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(title="保存结果", defaultextension=".tif", filetypes=filetypes)
        if not filepath:
            return

        try:
            if filepath.endswith('.tif') or filepath.endswith('.tiff'):
                geo_transform, crs = self.get_geo_transform()
                with rasterio.open(
                        filepath,
                        'w',
                        driver='GTiff',
                        height=self.result_data.shape[0],
                        width=self.result_data.shape[1],
                        count=1,
                        dtype=self.result_data.dtype,
                        crs=crs,
                        transform=geo_transform,
                ) as dst:
                    dst.write(self.result_data, 1)
            elif filepath.endswith('.jpg') or filepath.endswith('.png'):
                if self.result_image:
                    self.result_image.save(filepath)
                elif self.result_data is not None:
                    image = Image.fromarray(
                        (self.result_data * 255 / (np.max(self.result_data) + 1e-6)).astype(np.uint8))
                    image.save(filepath)
                else:
                    messagebox.showerror("错误", "无法保存结果为图像格式。")
            else:
                messagebox.showerror("错误", "不支持的文件格式。")
            messagebox.showinfo("保存成功", f"结果已保存到 {filepath}")
        except Exception as e:
            logging.error("保存结果失败", exc_info=True)
            messagebox.showerror("错误", f"保存结果失败: {e}")

    def create_progress_bar(self, title):
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title(title)
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.progress_window.protocol("WM_DELETE_WINDOW", self.disable_event)

    def disable_event(self):
        pass  # 禁用关闭窗口

    def train_sracn_model(self, model, train_loader, criterion_class, optimizer, num_epochs, device, label_mapping):
        try:
            total_steps = len(train_loader) * num_epochs
            step = 0

            for epoch in range(num_epochs):
                model.train()
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs, _ = model(inputs)
                    loss = criterion_class(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    step += 1
                    progress = (step / total_steps) * 100
                    self.progress_var.set(progress)
                    self.progress_window.update()

            self.progress_window.destroy()

            # 训练完成，开始预测
            self.predict_with_sracn_model(model, device, label_mapping)
        except Exception as e:
            logging.error("训练模型失败", exc_info=True)
            messagebox.showerror("错误", f"训练模型失败: {e}")
            self.progress_window.destroy()
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def predict_with_sracn_model(self, model, device, label_mapping):
        try:
            # 提取影像数据并进行预测
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
            spectral_mean = data_reshaped.mean(axis=0)
            spectral_std = data_reshaped.std(axis=0) + 1e-8
            data_normalized = (data_reshaped - spectral_mean) / spectral_std
            data_normalized = np.clip(data_normalized, -5, 5)
            data_tensor = torch.tensor(data_normalized, dtype=torch.float32).to(device)

            # 创建进度条窗口
            self.create_progress_bar("模型预测中...")

            batch_size = 1024
            num_samples = data_tensor.shape[0]
            predictions = []

            model.eval()
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    inputs = data_tensor[i:i+batch_size]
                    outputs, _ = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predictions.append(predicted.cpu().numpy())

                    progress = (i / num_samples) * 100
                    self.progress_var.set(progress)
                    self.progress_window.update()

            self.progress_window.destroy()

            predictions = np.concatenate(predictions)
            prediction_map = predictions.reshape(self.image_data.shape[0], self.image_data.shape[1])

            # 显示预测结果
            num_classes = len(label_mapping)
            color_map = plt.get_cmap('tab20', num_classes)
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(prediction_map, cmap=color_map, vmin=0, vmax=num_classes-1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            cbar = fig.colorbar(cax, ax=ax, ticks=range(num_classes))
            cbar_labels = [label for label, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
            cbar.ax.set_yticklabels(cbar_labels)
            ax.set_title("SRACN 模型预测结果")

            for widget in self.display_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.result_data = prediction_map
            self.result_image = None

        except Exception as e:
            logging.error("模型预测失败", exc_info=True)
            messagebox.showerror("错误", f"模型预测失败: {e}")
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def save_result(self):
        if self.result_data is None and self.result_image is None:
            messagebox.showerror("错误", "没有可保存的结果。")
            return

        filetypes = [
            ("GeoTIFF files", "*.tif"),
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(title="保存结果", defaultextension=".tif", filetypes=filetypes)
        if not filepath:
            return

        try:
            if filepath.endswith('.tif') or filepath.endswith('.tiff'):
                geo_transform, crs = self.get_geo_transform()
                with rasterio.open(
                    filepath,
                    'w',
                    driver='GTiff',
                    height=self.result_data.shape[0],
                    width=self.result_data.shape[1],
                    count=1,
                    dtype=self.result_data.dtype,
                    crs=crs,
                    transform=geo_transform,
                ) as dst:
                    dst.write(self.result_data, 1)
            elif filepath.endswith('.jpg') or filepath.endswith('.png'):
                if self.result_image:
                    self.result_image.save(filepath)
                elif self.result_data is not None:
                    image = Image.fromarray((self.result_data * 255 / (np.max(self.result_data) + 1e-6)).astype(np.uint8))
                    image.save(filepath)
                else:
                    messagebox.showerror("错误", "无法保存结果为图像格式。")
            else:
                messagebox.showerror("错误", "不支持的文件格式。")
            messagebox.showinfo("保存成功", f"结果已保存到 {filepath}")
        except Exception as e:
            logging.error("保存结果失败", exc_info=True)
            messagebox.showerror("错误", f"保存结果失败: {e}")

    def run_sracn_model(self):
        if not self.labels:
            messagebox.showerror("错误", "请先标注像元以获取训练样本。")
            return

        num_epochs = self.num_epochs.get()

        # 提取标注的光谱和标签
        spectra = []
        labels = []
        label_mapping = {}
        label_counter = 0

        for (x, y), label in self.labels.items():
            spectrum = self.image_data[y, x, :]
            spectra.append(spectrum)

            if label not in label_mapping:
                label_mapping[label] = label_counter
                label_counter += 1

            labels.append(label_mapping[label])

        spectra = np.array(spectra)
        labels = np.array(labels)

        # 创建数据集
        dataset = MineralDataset(spectra, labels)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

        # 定义模型
        input_length = spectra.shape[1]
        num_classes = len(label_mapping)
        model = SpectralResidualAttentionConvolutionalNetwork(
            input_length=input_length,
            num_classes=num_classes
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        criterion_class = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # 创建进度条窗口
        self.create_progress_bar("训练模型中...")

        # 启动后台线程进行训练
        thread = threading.Thread(target=self.train_sracn_model,
                                  args=(model, train_loader, criterion_class, optimizer, num_epochs, device, label_mapping))
        thread.start()

    def create_progress_bar(self, title):
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title(title)
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.progress_window.protocol("WM_DELETE_WINDOW", self.disable_event)

    def disable_event(self):
        pass  # 禁用关闭窗口

    def train_sracn_model(self, model, train_loader, criterion_class, optimizer, num_epochs, device, label_mapping):
        try:
            total_steps = len(train_loader) * num_epochs
            step = 0

            for epoch in range(num_epochs):
                model.train()
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs, _ = model(inputs)
                    loss = criterion_class(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    step += 1
                    progress = (step / total_steps) * 100
                    self.progress_var.set(progress)
                    self.progress_window.update()

            self.progress_window.destroy()

            # 训练完成，开始预测
            self.predict_with_sracn_model(model, device, label_mapping)
        except Exception as e:
            logging.error("训练模型失败", exc_info=True)
            messagebox.showerror("错误", f"训练模型失败: {e}")
            self.progress_window.destroy()
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def predict_with_sracn_model(self, model, device, label_mapping):
        try:
            # 提取影像数据并进行预测
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
            spectral_mean = data_reshaped.mean(axis=0)
            spectral_std = data_reshaped.std(axis=0) + 1e-8
            data_normalized = (data_reshaped - spectral_mean) / spectral_std
            data_normalized = np.clip(data_normalized, -5, 5)
            data_tensor = torch.tensor(data_normalized, dtype=torch.float32).to(device)

            # 创建进度条窗口
            self.create_progress_bar("模型预测中...")

            batch_size = 1024
            num_samples = data_tensor.shape[0]
            predictions = []

            model.eval()
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    inputs = data_tensor[i:i+batch_size]
                    outputs, _ = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predictions.append(predicted.cpu().numpy())

                    progress = (i / num_samples) * 100
                    self.progress_var.set(progress)
                    self.progress_window.update()

            self.progress_window.destroy()

            predictions = np.concatenate(predictions)
            prediction_map = predictions.reshape(self.image_data.shape[0], self.image_data.shape[1])

            # 显示预测结果
            num_classes = len(label_mapping)
            color_map = plt.get_cmap('tab20', num_classes)
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(prediction_map, cmap=color_map, vmin=0, vmax=num_classes-1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            cbar = fig.colorbar(cax, ax=ax, ticks=range(num_classes))
            cbar_labels = [label for label, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
            cbar.ax.set_yticklabels(cbar_labels)
            ax.set_title("SRACN 模型预测结果")

            for widget in self.display_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.result_data = prediction_map
            self.result_image = None

        except Exception as e:
            logging.error("模型预测失败", exc_info=True)
            messagebox.showerror("错误", f"模型预测失败: {e}")
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def save_result(self):
        if self.result_data is None and self.result_image is None:
            messagebox.showerror("错误", "没有可保存的结果。")
            return

        filetypes = [
            ("GeoTIFF files", "*.tif"),
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(title="保存结果", defaultextension=".tif", filetypes=filetypes)
        if not filepath:
            return

        try:
            if filepath.endswith('.tif') or filepath.endswith('.tiff'):
                geo_transform, crs = self.get_geo_transform()
                with rasterio.open(
                    filepath,
                    'w',
                    driver='GTiff',
                    height=self.result_data.shape[0],
                    width=self.result_data.shape[1],
                    count=1,
                    dtype=self.result_data.dtype,
                    crs=crs,
                    transform=geo_transform,
                ) as dst:
                    dst.write(self.result_data, 1)
            elif filepath.endswith('.jpg') or filepath.endswith('.png'):
                if self.result_image:
                    self.result_image.save(filepath)
                elif self.result_data is not None:
                    image = Image.fromarray((self.result_data * 255 / (np.max(self.result_data) + 1e-6)).astype(np.uint8))
                    image.save(filepath)
                else:
                    messagebox.showerror("错误", "无法保存结果为图像格式。")
            else:
                messagebox.showerror("错误", "不支持的文件格式。")
            messagebox.showinfo("保存成功", f"结果已保存到 {filepath}")
        except Exception as e:
            logging.error("保存结果失败", exc_info=True)
            messagebox.showerror("错误", f"保存结果失败: {e}")

    def run_sracn_model(self):
        if not self.labels:
            messagebox.showerror("错误", "请先标注像元以获取训练样本。")
            return

        num_epochs = self.num_epochs.get()

        # 提取标注的光谱和标签
        spectra = []
        labels = []
        label_mapping = {}
        label_counter = 0

        for (x, y), label in self.labels.items():
            spectrum = self.image_data[y, x, :]
            spectra.append(spectrum)

            if label not in label_mapping:
                label_mapping[label] = label_counter
                label_counter += 1

            labels.append(label_mapping[label])

        spectra = np.array(spectra)
        labels = np.array(labels)

        # 创建数据集
        dataset = MineralDataset(spectra, labels)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

        # 定义模型
        input_length = spectra.shape[1]
        num_classes = len(label_mapping)
        model = SpectralResidualAttentionConvolutionalNetwork(
            input_length=input_length,
            num_classes=num_classes
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        criterion_class = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # 创建进度条窗口
        self.create_progress_bar("训练模型中...")

        # 启动后台线程进行训练
        thread = threading.Thread(target=self.train_sracn_model,
                                  args=(model, train_loader, criterion_class, optimizer, num_epochs, device, label_mapping))
        thread.start()

    def create_progress_bar(self, title):
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title(title)
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.progress_window.protocol("WM_DELETE_WINDOW", self.disable_event)

    def disable_event(self):
        pass  # 禁用关闭窗口

    def train_sracn_model(self, model, train_loader, criterion_class, optimizer, num_epochs, device, label_mapping):
        try:
            total_steps = len(train_loader) * num_epochs
            step = 0

            for epoch in range(num_epochs):
                model.train()
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs, _ = model(inputs)
                    loss = criterion_class(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    step += 1
                    progress = (step / total_steps) * 100
                    self.progress_var.set(progress)
                    self.progress_window.update()

            self.progress_window.destroy()

            # 训练完成，开始预测
            self.predict_with_sracn_model(model, device, label_mapping)
        except Exception as e:
            logging.error("训练模型失败", exc_info=True)
            messagebox.showerror("错误", f"训练模型失败: {e}")
            self.progress_window.destroy()
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def predict_with_sracn_model(self, model, device, label_mapping):
        try:
            # 提取影像数据并进行预测
            data_reshaped = self.image_data.reshape(-1, self.image_data.shape[2])
            spectral_mean = data_reshaped.mean(axis=0)
            spectral_std = data_reshaped.std(axis=0) + 1e-8
            data_normalized = (data_reshaped - spectral_mean) / spectral_std
            data_normalized = np.clip(data_normalized, -5, 5)
            data_tensor = torch.tensor(data_normalized, dtype=torch.float32).to(device)

            # 创建进度条窗口
            self.create_progress_bar("模型预测中...")

            batch_size = 1024
            num_samples = data_tensor.shape[0]
            predictions = []

            model.eval()
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    inputs = data_tensor[i:i+batch_size]
                    outputs, _ = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predictions.append(predicted.cpu().numpy())

                    progress = (i / num_samples) * 100
                    self.progress_var.set(progress)
                    self.progress_window.update()

            self.progress_window.destroy()

            predictions = np.concatenate(predictions)
            prediction_map = predictions.reshape(self.image_data.shape[0], self.image_data.shape[1])

            # 显示预测结果
            num_classes = len(label_mapping)
            color_map = plt.get_cmap('tab20', num_classes)
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(prediction_map, cmap=color_map, vmin=0, vmax=num_classes-1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            cbar = fig.colorbar(cax, ax=ax, ticks=range(num_classes))
            cbar_labels = [label for label, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
            cbar.ax.set_yticklabels(cbar_labels)
            ax.set_title("SRACN 模型预测结果")

            for widget in self.display_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.result_data = prediction_map
            self.result_image = None

        except Exception as e:
            logging.error("模型预测失败", exc_info=True)
            messagebox.showerror("错误", f"模型预测失败: {e}")
        finally:
            # 启用分析按钮
            self.run_analysis_button.config(state=tk.NORMAL)

    def save_result(self):
        if self.result_data is None and self.result_image is None:
            messagebox.showerror("错误", "没有可保存的结果。")
            return

        filetypes = [
            ("GeoTIFF files", "*.tif"),
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(title="保存结果", defaultextension=".tif", filetypes=filetypes)
        if not filepath:
            return

        try:
            if filepath.endswith('.tif') or filepath.endswith('.tiff'):
                geo_transform, crs = self.get_geo_transform()
                with rasterio.open(
                    filepath,
                    'w',
                    driver='GTiff',
                    height=self.result_data.shape[0],
                    width=self.result_data.shape[1],
                    count=1,
                    dtype=self.result_data.dtype,
                    crs=crs,
                    transform=geo_transform,
                ) as dst:
                    dst.write(self.result_data, 1)
            elif filepath.endswith('.jpg') or filepath.endswith('.png'):
                if self.result_image:
                    self.result_image.save(filepath)
                elif self.result_data is not None:
                    image = Image.fromarray((self.result_data * 255 / (np.max(self.result_data) + 1e-6)).astype(np.uint8))
                    image.save(filepath)
                else:
                    messagebox.showerror("错误", "无法保存结果为图像格式。")
            else:
                messagebox.showerror("错误", "不支持的文件格式。")
            messagebox.showinfo("保存成功", f"结果已保存到 {filepath}")
        except Exception as e:
            logging.error("保存结果失败", exc_info=True)
            messagebox.showerror("错误", f"保存结果失败: {e}")
    def update_label_scaling(self):
        # 更新所有标注的缩放比例
        for (x, y), label in self.labels.items():
            # 假设标签是用矩形标记，这里重新绘制
            # 您可以根据需要调整标记方式
            color = self.get_label_color(label)
            self.canvas.create_rectangle(
                x * self.zoom_level, y * self.zoom_level,
                (x + 1) * self.zoom_level, (y + 1) * self.zoom_level,
                outline=color
            )

    def get_label_color(self, label=None):
        # 获取标签的颜色
        if label:
            if label not in self.label_colors:
                self.label_colors[label] = self.colors_list[self.color_index % len(self.colors_list)]
                self.color_index += 1
            return self.label_colors[label]
        else:
            # 默认颜色
            return 'red'

    def get_next_color(self):
        # 获取下一个颜色，用于绘制不同像素的光谱
        color = self.colors_list[self.color_index % len(self.colors_list)]
        self.color_index += 1
        return color

    def save_spectrum(self):
        if not hasattr(self, 'spectrum_data') or not self.spectrum_data:
            messagebox.showerror("错误", "没有波谱数据可保存。")
            return

        filepath = filedialog.asksaveasfilename(title="保存波谱为TXT", defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                f.write("波长 (nm)\t反射率\n")
                for label, (wavelengths, reflectance, color) in self.spectrum_data.items():
                    for wl, refl in zip(wavelengths, reflectance):
                        f.write(f"{wl}\t{refl}\n")
            messagebox.showinfo("保存成功", f"波谱已保存到 {filepath}")
        except Exception as e:
            logging.error("保存波谱失败", exc_info=True)
            messagebox.showerror("错误", f"保存波谱失败: {e}")

    def update_legend(self):
        # 更新图例，显示不同标签对应的颜色
        if not hasattr(self, 'legend_fig') or self.legend_fig is None:
            self.legend_fig, self.legend_ax = plt.subplots(figsize=(2, 6))
            self.legend_canvas = FigureCanvasTkAgg(self.legend_fig, master=self.display_frame)
            self.legend_canvas.draw()
            self.legend_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.legend_ax.clear()

        handles = []
        labels = []
        for label, color in self.label_colors.items():
            handles.append(plt.Line2D([0], [0], color=color, lw=4))
            labels.append(label)

        self.legend_ax.legend(handles, labels, loc='upper left')
        self.legend_ax.axis('off')
        self.legend_canvas.draw()

if __name__ == "__main__":
    try:
            multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置了启动方法

    try:
        root = tk.Tk()
        app = ENVIStyleViewerApp(root)
        root.mainloop()
    except Exception as e:
        logging.error("程序出错", exc_info=True)
        print(f"程序出错: {e}")


