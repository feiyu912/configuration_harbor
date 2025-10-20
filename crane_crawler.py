import os
import re
import time
import hashlib
import requests
from urllib import error
from bs4 import BeautifulSoup
from pathlib import Path
import imagehash
from PIL import Image
import concurrent.futures
from tqdm import tqdm

class CraneCrawler:
    def __init__(self, save_dir="datasets/private_ship"):
        self.save_dir = Path(save_dir)
        self.images_dir = self.save_dir / "JPEGImages"
        self.annotations_dir = self.save_dir / "Annotations"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        self.downloaded_hashes = set()
        self.failed_urls = []

    def is_good_image(self, img_path, min_size=(200, 200), max_size=(4000, 4000)):
        """检查图片质量"""
        try:
            with Image.open(img_path) as img:
                # 检查尺寸
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    return False
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    return False

                # 检查宽高比（避免极端比例）
                ratio = max(img.size) / min(img.size)
                if ratio > 5:
                    return False

                return True
        except:
            return False

    def is_duplicate(self, img_path):
        """检查是否重复图片"""
        try:
            with Image.open(img_path) as img:
                # 计算感知哈希
                img_hash = str(imagehash.phash(img))
                if img_hash in self.downloaded_hashes:
                    return True
                self.downloaded_hashes.add(img_hash)
                return False
        except:
            return True

    def download_image(self, url, filename, max_retries=3):
        """下载单张图片"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10, stream=True)
                response.raise_for_status()

                # 检查内容类型
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    return False

                # 保存图片
                img_path = self.images_dir / filename
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # 质量检查
                if not self.is_good_image(img_path):
                    img_path.unlink()
                    return False

                # 重复检查
                if self.is_duplicate(img_path):
                    img_path.unlink()
                    return False

                return True

            except Exception as e:
                if attempt == max_retries - 1:
                    self.failed_urls.append(url)
                time.sleep(1)

        return False

    def crawl_baidu_images(self, keywords, max_images=1000):
        """爬取百度图片"""
        print(f"开始爬取船图片，目标数量: {max_images}")

        downloaded = 0
        for keyword in keywords:
            if downloaded >= max_images:
                break

            print(f"搜索关键词: {keyword}")
            url_template = f'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={keyword}&pn='

            page = 0
            while downloaded < max_images:
                try:
                    url = url_template + str(page * 60)
                    response = self.session.get(url, timeout=10)
                    response.encoding = 'utf-8'

                    # 提取图片URL
                    pic_urls = re.findall('"objURL":"(.*?)"', response.text)

                    if not pic_urls:
                        print(f"关键词 {keyword} 搜索完毕")
                        break

                    # 并发下载
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        futures = []
                        for i, pic_url in enumerate(pic_urls[:10]):  # 每页最多10张
                            if downloaded >= max_images:
                                break
                            filename = f"船_{downloaded:06d}.jpg"
                            future = executor.submit(self.download_image, pic_url, filename)
                            futures.append(future)
                            downloaded += 1

                        # 等待完成
                        for future in tqdm(concurrent.futures.as_completed(futures),
                                         desc=f"下载 {keyword}", total=len(futures)):
                            if future.result():
                                pass  # 成功下载

                    page += 1
                    time.sleep(2)  # 避免请求过快

                except Exception as e:
                    print(f"搜索 {keyword} 时出错: {e}")
                    break

        print(f"爬取完成！成功下载 {len(list(self.images_dir.glob('*.jpg')))} 张图片")
        return len(list(self.images_dir.glob('*.jpg')))





def main():
    # 自制吊机关键词（与公开的不同类型）
    keywords = [
        "港口船",

    ]

    crawler = CraneCrawler()
    count = crawler.crawl_baidu_images(keywords, max_images=700)

    print(f"爬取完成，共获得 {count} 张图片")
    print(f"图片保存在: {crawler.images_dir}")
    print(f"失败URL数量: {len(crawler.failed_urls)}")

if __name__ == "__main__":
    main()
