import tkinter as tk
from tkinter import simpledialog, filedialog
from PIL import Image, ImageTk
import math
import random 

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Programı")

        button_frame = tk.Frame(root)
        button_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.original_image_label = tk.Label(root)
        self.original_image_label.grid(row=0, column=0, rowspan=12, padx=10, pady=10)

        self.processed_image_label = tk.Label(root)
        self.processed_image_label.grid(row=0, column=2, rowspan=12, padx=10, pady=10)

        self.load_button = tk.Button(button_frame, text="Resim Yükle", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.load_second_button = tk.Button(button_frame, text="İkinci Resmi Yükle", command=self.load_second_image)
        self.load_second_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        button_texts = ["Gri Dönüşüm", "Binarization", "Döndür", "Kırp", "Yakınlaştır/Uzaklaştır", 
                        "Renk Uzayı Dönüşümleri", "Histogram Germe", "Resim Ekleme", "Resim Çarpma", 
                        "Parlaklık Artır", "Konvolüsyon (Gauss)", "Adaptif Eşikleme", "Kenar Bulma (Sobel)", 
                        "Gürültü Ekle (Salt&Pepper) ve Temizle (Mean, Median)", "Görüntüye Filtre Uygulanması (Blurring)", 
                        "Morfolojik İşlemler (Genişleme, Aşınma, Açma, Kapama)"]

        row_num = 2
        for text in button_texts:
            button = tk.Button(button_frame, text=text, command=lambda t=text: self.process_image(button_texts.index(t) + 3))
            button.grid(row=row_num, column=0, padx=10, pady=5, sticky="ew")
            row_num += 1

        self.original_image = None
        self.second_image = None
        self.processed_image = None
        self.photo = None
        self.processed_photo = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.original_image = Image.open(file_path).convert("RGB")
            self.processed_image = self.original_image.copy()
            self.photo = ImageTk.PhotoImage(self.original_image)
            self.original_image_label.config(image=self.photo)
            self.processed_photo = ImageTk.PhotoImage(self.processed_image)
            self.processed_image_label.config(image=self.processed_photo)

    def load_second_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.second_image = Image.open(file_path).convert("RGB")


    def process_image(self, index):
        if self.original_image:
            if index == 3:
                self.processed_image = self.to_grayscale(self.original_image)
            elif index == 4:
                self.processed_image = self.binarize(self.original_image)
            elif index == 5:
                angle = simpledialog.askinteger("Döndür", "Açıyı girin:", minvalue=0, maxvalue=360)
                if angle is not None:
                    self.processed_image = self.rotate_image(self.original_image, angle)
            elif index == 6:
                start_x = simpledialog.askinteger("Kırpma", "Başlangıç X koordinatını girin:", minvalue=0, maxvalue=self.original_image.width)
                start_y = simpledialog.askinteger("Kırpma", "Başlangıç Y koordinatını girin:", minvalue=0, maxvalue=self.original_image.height)
                end_x = simpledialog.askinteger("Kırpma", "Bitiş X koordinatını girin:", minvalue=0, maxvalue=self.original_image.width)
                end_y = simpledialog.askinteger("Kırpma", "Bitiş Y koordinatını girin:", minvalue=0, maxvalue=self.original_image.height)
                if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
                    self.processed_image = self.crop_image(self.original_image, start_x, start_y, end_x, end_y)
            elif index == 7:
                scale_factor = simpledialog.askfloat("Yakınlaştır/Uzaklaştır", "Ölçek faktörünü girin (örn. 2, 0.5):", minvalue=0.1)
                if scale_factor is not None:
                    self.processed_image = self.resize_image(self.original_image, scale_factor)
            elif index == 8:
                self.select_color_space_conversion()
            elif index == 9:
                self.processed_image = self.histogram_stretch(self.original_image)
            elif index == 10:
                if self.second_image:
                    self.processed_image = self.add_images(self.original_image, self.second_image)
            elif index == 11:
                if self.second_image:
                    self.processed_image = self.multiply_images(self.original_image, self.second_image)
            elif index == 12:
                brightness_increase = simpledialog.askinteger("Parlaklık Artır", "Parlaklık artırma miktarını girin:", minvalue=0, maxvalue=255)
                if brightness_increase is not None:
                    self.processed_image = self.increase_brightness(self.original_image, brightness_increase)
            elif index == 13:
                kernel_size = simpledialog.askinteger("Gauss Konvolüsyon", "Kernel boyutunu girin (örn. 3, 5, 7):", minvalue=3)
                sigma = simpledialog.askfloat("Gauss Konvolüsyon", "Sigma değerini girin (örn. 1.0, 2.0):", minvalue=0.1)
                if kernel_size and sigma:
                    self.processed_image = self.gaussian_convolution(self.original_image, kernel_size, sigma)
            elif index == 14:
                block_size = simpledialog.askinteger("Adaptif Eşikleme", "Blok boyutunu girin:", minvalue=3)
                c = simpledialog.askinteger("Adaptif Eşikleme", "C değerini girin:", minvalue=0)
                if block_size and c is not None:
                    self.processed_image = self.adaptive_thresholding(self.original_image, block_size, c)
            elif index == 15:
                self.processed_image = self.sobel_edge_detection(self.original_image)
            elif index == 16:
                noise_amount = simpledialog.askfloat("Gürültü Ekle (Salt&Pepper)", "Gürültü miktarını girin (0-1 arası):", minvalue=0.0, maxvalue=1.0)
                if noise_amount is not None:
                    self.processed_image = self.add_salt_pepper_noise(self.original_image, noise_amount)
                    filter_choice = simpledialog.askstring("Filtre Seçimi", "Filtre türünü girin (mean, median):")
                    if filter_choice:
                        if filter_choice.lower() == "mean":
                            self.processed_image = self.mean_filter(self.processed_image)
                        elif filter_choice.lower() == "median":
                            self.processed_image = self.median_filter(self.processed_image)
            elif index == 17:
                kernel_size = simpledialog.askinteger("Blurring", "Kernel boyutunu girin (örn. 3, 5, 7):", minvalue=3)
                if kernel_size:
                    self.processed_image = self.blur_image(self.original_image, kernel_size)
            elif index == 18:
                self.select_morphological_operation()

            self.processed_photo = ImageTk.PhotoImage(self.processed_image)
            self.processed_image_label.config(image=self.processed_photo)



    def to_grayscale(self, img):
        grayscale_img = Image.new("L", img.size)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                gray = int(0.299*r + 0.587*g + 0.114*b)
                grayscale_img.putpixel((x, y), gray)
        return grayscale_img.convert("RGB")

    def binarize(self, img, threshold=128):
        binary_img = Image.new("1", img.size)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                gray = int(0.299*r + 0.587*g + 0.114*b)
                binary = 255 if gray > threshold else 0
                binary_img.putpixel((x, y), binary)
        return binary_img.convert("RGB")

    def rotate_image(self, img, angle):
        width, height = img.size
        angle_rad = math.radians(angle)
        sin_angle = math.sin(angle_rad)
        cos_angle = math.cos(angle_rad)

        new_width = int(abs(width * cos_angle) + abs(height * sin_angle))
        new_height = int(abs(width * sin_angle) + abs(height * cos_angle))

        rotated_img = Image.new("RGB", (new_width, new_height))
        original_center_x = width / 2
        original_center_y = height / 2
        new_center_x = new_width / 2
        new_center_y = new_height / 2

        for x in range(new_width):
            for y in range(new_height):
                original_x = cos_angle * (x - new_center_x) + sin_angle * (y - new_center_y) + original_center_x
                original_y = -sin_angle * (x - new_center_x) + cos_angle * (y - new_center_y) + original_center_y
                if 0 <= original_x < width and 0 <= original_y < height:
                    rotated_img.putpixel((x, y), img.getpixel((int(original_x), int(original_y))))
        
        return rotated_img

    def crop_image(self, img, start_x, start_y, end_x, end_y):
        width, height = img.size
        cropped_width = end_x - start_x
        cropped_height = end_y - start_y
        cropped_img = Image.new("RGB", (cropped_width, cropped_height))
        for x in range(cropped_width):
            for y in range(cropped_height):
                cropped_img.putpixel((x, y), img.getpixel((x + start_x, y + start_y)))
        return cropped_img

    def resize_image(self, img, scale_factor):
        width, height = img.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_img = Image.new("RGB", (new_width, new_height))
        for x in range(new_width):
            for y in range(new_height):
                src_x = int(x / scale_factor)
                src_y = int(y / scale_factor)
                resized_img.putpixel((x, y), img.getpixel((src_x, src_y)))
        return resized_img

    def select_color_space_conversion(self):
        def apply_conversion(choice):
            if choice == "HSV Dönüşümü":
                self.processed_image = self.rgb_to_hsv(self.original_image)
            elif choice == "BGR Dönüşümü":
                self.processed_image = self.rgb_to_bgr(self.original_image)
            elif choice == "LUV Dönüşümü":
                self.processed_image = self.rgb_to_luv(self.original_image)
            self.processed_photo = ImageTk.PhotoImage(self.processed_image)
            self.processed_image_label.config(image=self.processed_photo)
            color_space_window.destroy()

        color_space_window = tk.Toplevel(self.root)
        color_space_window.title("Renk Uzayı Dönüşümü Seç")
        options = ["HSV Dönüşümü", "BGR Dönüşümü", "LUV Dönüşümü"]
        choice = tk.StringVar()
        choice.set(options[0])
        for option in options:
            rb = tk.Radiobutton(color_space_window, text=option, variable=choice, value=option)
            rb.pack(anchor=tk.W)
        apply_button = tk.Button(color_space_window, text="Uygula", command=lambda: apply_conversion(choice.get()))
        apply_button.pack()

    def rgb_to_hsv(self, img):
        hsv_img = Image.new("RGB", img.size)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                r, g, b = r / 255.0, g / 255.0, b / 255.0
                mx = max(r, g, b)
                mn = min(r, g, b)
                df = mx - mn
                if mx == mn:
                    h = 0
                elif mx == r:
                    h = (60 * ((g - b) / df) + 360) % 360
                elif mx == g:
                    h = (60 * ((b - r) / df) + 120) % 360
                elif mx == b:
                    h = (60 * ((r - g) / df) + 240) % 360
                if mx == 0:
                    s = 0
                else:
                    s = df / mx
                v = mx
                hsv_img.putpixel((x, y), (int(h / 360 * 255), int(s * 255), int(v * 255)))
        return hsv_img

    def rgb_to_bgr(self, img):
        bgr_img = Image.new("RGB", img.size)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                bgr_img.putpixel((x, y), (b, g, r))
        return bgr_img

    def rgb_to_luv(self, img):
        luv_img = Image.new("RGB", img.size)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                l = 0.299 * r + 0.587 * g + 0.114 * b
                denominator = r + 15 * g + 3 * b
                u = 0 if denominator == 0 else 1 / (1 + 15 * (4 * r / denominator))
                v = 0 if denominator == 0 else 1 / (1 + 15 * (9 * g / denominator))
                luv_img.putpixel((x, y), (int(l), int(u * 255), int(v * 255)))
        return luv_img

    def histogram_stretch(self, img):
        width, height = img.size
        r_min, r_max, g_min, g_max, b_min, b_max = 255, 0, 255, 0, 255, 0

        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                r_min, r_max = min(r_min, r), max(r_max, r)
                g_min, g_max = min(g_min, g), max(g_max, g)
                b_min, b_max = min(b_min, b), max(b_max, b)

        def stretch(val, min_val, max_val):
            return int((val - min_val) / (max_val - min_val) * 255)

        stretched_img = Image.new("RGB", (width, height))
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                r_stretched = stretch(r, r_min, r_max)
                g_stretched = stretch(g, g_min, g_max)
                b_stretched = stretch(b, b_min, b_max)
                stretched_img.putpixel((x, y), (r_stretched, g_stretched, b_stretched))

        return stretched_img
    
    def add_images(self, img1, img2):
        width1, height1 = img1.size
        width2, height2 = img2.size
        min_width = min(width1, width2)
        min_height = min(height1, height2)
        added_img = Image.new("RGB", (min_width, min_height))
        for x in range(min_width):
            for y in range(min_height):
                r1, g1, b1 = img1.getpixel((x, y))
                r2, g2, b2 = img2.getpixel((x, y))
                r_sum = min(r1 + r2, 255)
                g_sum = min(g1 + g2, 255)
                b_sum = min(b1 + b2, 255)
                added_img.putpixel((x, y), (r_sum, g_sum, b_sum))
        return added_img
    
    def multiply_images(self, img1, img2):
        width1, height1 = img1.size
        width2, height2 = img2.size
        min_width = min(width1, width2)
        min_height = min(height1, height2)
        multiplied_img = Image.new("RGB", (min_width, min_height))
        for x in range(min_width):
            for y in range(min_height):
                r1, g1, b1 = img1.getpixel((x, y))
                r2, g2, b2 = img2.getpixel((x, y))
                r_mult = min(r1 * r2 // 255, 255)
                g_mult = min(g1 * g2 // 255, 255)
                b_mult = min(b1 * b2 // 255, 255)
                multiplied_img.putpixel((x, y), (r_mult, g_mult, b_mult))
        return multiplied_img

    def increase_brightness(self, img, increase):
        bright_img = Image.new("RGB", img.size)
        width, height = img.size
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                r = min(r + increase, 255)
                g = min(g + increase, 255)
                b = min(b + increase, 255)
                bright_img.putpixel((x, y), (r, g, b))
        return bright_img
    
    def gaussian_convolution(self, img, kernel_size, sigma):
        def gauss(x, y, sigma):
            return (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        
        kernel = []
        kernel_sum = 0
        offset = kernel_size // 2
        for x in range(-offset, offset + 1):
            row = []
            for y in range(-offset, offset + 1):
                value = gauss(x, y, sigma)
                row.append(value)
                kernel_sum += value
            kernel.append(row)
        
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[x][y] /= kernel_sum
        
        width, height = img.size
        convolved_img = Image.new("RGB", img.size)
        
        for x in range(width):
            for y in range(height):
                r_sum, g_sum, b_sum = 0, 0, 0
                for kx in range(-offset, offset + 1):
                    for ky in range(-offset, offset + 1):
                        px = min(max(x + kx, 0), width - 1)
                        py = min(max(y + ky, 0), height - 1)
                        r, g, b = img.getpixel((px, py))
                        weight = kernel[kx + offset][ky + offset]
                        r_sum += r * weight
                        g_sum += g * weight
                        b_sum += b * weight
                convolved_img.putpixel((x, y), (int(r_sum), int(g_sum), int(b_sum)))
        
        return convolved_img

    def adaptive_thresholding(self, img, block_size, c):
        grayscale_img = self.to_grayscale(img)
        width, height = grayscale_img.size
        thresholded_img = Image.new("1", (width, height))
        for x in range(width):
            for y in range(height):
                x_start = max(0, x - block_size // 2)
                y_start = max(0, y - block_size // 2)
                x_end = min(width, x + block_size // 2)
                y_end = min(height, y + block_size // 2)
                block_pixels = []
                for i in range(x_start, x_end):
                    for j in range(y_start, y_end):
                        block_pixels.append(grayscale_img.getpixel((i, j))[0])
                block_mean = sum(block_pixels) / len(block_pixels)
                threshold = block_mean - c
                gray = grayscale_img.getpixel((x, y))[0]
                thresholded_img.putpixel((x, y), 255 if gray > threshold else 0)
        return thresholded_img.convert("RGB")
    
    def sobel_edge_detection(self, img):
        def apply_kernel(img, kernel):
            width, height = img.size
            result_img = Image.new("L", (width, height))
            offset = len(kernel) // 2
            for x in range(offset, width - offset):
                for y in range(offset, height - offset):
                    pixel_value = 0
                    for kx in range(len(kernel)):
                        for ky in range(len(kernel)):
                            px, py = x + kx - offset, y + ky - offset
                            pixel_value += img.getpixel((px, py)) * kernel[kx][ky]
                    result_img.putpixel((x, y), int(pixel_value))
            return result_img

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        gray_img = self.to_grayscale(img).convert("L")
        grad_x = apply_kernel(gray_img, sobel_x)
        grad_y = apply_kernel(gray_img, sobel_y)

        width, height = img.size
        edge_img = Image.new("L", img.size)
        for x in range(width):
            for y in range(height):
                gx = grad_x.getpixel((x, y))
                gy = grad_y.getpixel((x, y))
                magnitude = int(math.sqrt(gx**2 + gy**2))
                edge_img.putpixel((x, y), magnitude)
        
        return edge_img.convert("RGB")
    
    def add_salt_pepper_noise(self, img, amount):
        noisy_img = img.copy()
        width, height = img.size
        num_noise_pixels = int(amount * width * height)
        for _ in range(num_noise_pixels):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if random.random() < 0.5:
                noisy_img.putpixel((x, y), (0, 0, 0))
            else:
                noisy_img.putpixel((x, y), (255, 255, 255))
        return noisy_img

    def mean_filter(self, img):
        width, height = img.size
        filtered_img = Image.new("RGB", img.size)
        kernel_size = 3
        offset = kernel_size // 2

        for x in range(offset, width - offset):
            for y in range(offset, height - offset):
                r_sum, g_sum, b_sum = 0, 0, 0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        xn = x + i - offset
                        yn = y + j - offset
                        r, g, b = img.getpixel((xn, yn))
                        r_sum += r
                        g_sum += g
                        b_sum += b
                r_avg = r_sum // (kernel_size ** 2)
                g_avg = g_sum // (kernel_size ** 2)
                b_avg = b_sum // (kernel_size ** 2)
                filtered_img.putpixel((x, y), (r_avg, g_avg, b_avg))
        return filtered_img

    def median_filter(self, img):
        width, height = img.size
        filtered_img = Image.new("RGB", img.size)
        kernel_size = 3
        offset = kernel_size // 2

        for x in range(offset, width - offset):
            for y in range(offset, height - offset):
                r_values, g_values, b_values = [], [], []
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        xn = x + i - offset
                        yn = y + j - offset
                        r, g, b = img.getpixel((xn, yn))
                        r_values.append(r)
                        g_values.append(g)
                        b_values.append(b)
                r_values.sort()
                g_values.sort()
                b_values.sort()
                r_median = r_values[len(r_values) // 2]
                g_median = g_values[len(g_values) // 2]
                b_median = b_values[len(b_values) // 2]
                filtered_img.putpixel((x, y), (r_median, g_median, b_median))
        return filtered_img
    
    def blur_image(self, img, kernel_size):
        width, height = img.size
        blurred_img = Image.new("RGB", img.size)
        offset = kernel_size // 2

        for x in range(width):
            for y in range(height):
                r_total, g_total, b_total = 0, 0, 0
                count = 0

                for dx in range(-offset, offset + 1):
                    for dy in range(-offset, offset + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            r, g, b = img.getpixel((nx, ny))
                            r_total += r
                            g_total += g
                            b_total += b
                            count += 1

                r_avg = r_total // count
                g_avg = g_total // count
                b_avg = b_total // count

                blurred_img.putpixel((x, y), (r_avg, g_avg, b_avg))

        return blurred_img
    
    def select_morphological_operation(self):
        operation = simpledialog.askstring("Morfolojik işlemler", "Morfolojik işlem seçiniz (Genişleme, Aşınma, Açma, Kapama):")
        if operation:
            if operation.lower() == "genişleme":
                radius = simpledialog.askinteger("Genişleme", "Genişleme yarıçapını giriniz:")
                if radius is not None:
                    self.processed_image = self.dilation(self.original_image, radius)
            elif operation.lower() == "aşınma":
                radius = simpledialog.askinteger("Aşınma", "Aşınma yarıçapını giriniz:")
                if radius is not None:
                    self.processed_image = self.erosion(self.original_image, radius)
            elif operation.lower() == "açma":
                radius = simpledialog.askinteger("Açma", "Açma yarıçapını giriniz:")
                if radius is not None:
                    self.processed_image = self.opening(self.original_image, radius)
            elif operation.lower() == "kapama":
                radius = simpledialog.askinteger("Kapama", "Kapama yarıçapını giriniz:")
                if radius is not None:
                    self.processed_image = self.closing(self.original_image, radius)

            self.processed_photo = ImageTk.PhotoImage(self.processed_image)
            self.processed_image_label.config(image=self.processed_photo)

    def dilation(self, img, radius):
        width, height = img.size
        dilated_img = Image.new("RGB", img.size)

        for x in range(width):
            for y in range(height):
                max_intensity = (0, 0, 0)
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx = min(max(x + dx, 0), width - 1)
                        ny = min(max(y + dy, 0), height - 1)
                        intensity = img.getpixel((nx, ny))
                        max_intensity = tuple(max(max_intensity[i], intensity[i]) for i in range(3))
                dilated_img.putpixel((x, y), max_intensity)

        return dilated_img

    def erosion(self, img, radius):
        width, height = img.size
        eroded_img = Image.new("RGB", img.size)

        for x in range(width):
            for y in range(height):
                min_intensity = (255, 255, 255)
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx = min(max(x + dx, 0), width - 1)
                        ny = min(max(y + dy, 0), height - 1)
                        intensity = img.getpixel((nx, ny))
                        min_intensity = tuple(min(min_intensity[i], intensity[i]) for i in range(3))
                eroded_img.putpixel((x, y), min_intensity)

        return eroded_img

    def opening(self, img, radius):
        temp_img = self.erosion(img, radius)
        opened_img = self.dilation(temp_img, radius)
        return opened_img

    def closing(self, img, radius):
        temp_img = self.dilation(img, radius)
        closed_img = self.erosion(temp_img, radius)
        return closed_img        



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
