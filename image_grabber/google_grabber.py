import json
import time
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from typing import List

from .abstract_grabber import AbstractGrabber
from .grabbed_image import GrabbedImage
from .grab_settings import *
from utils.utils import StringUtil


class GoogleGrabber(AbstractGrabber):
    """Grab images from google search"""

    full_image = True
    GOOGLE_URL = "https://www.google.co.in/search?q=%s&source=lnms&tbm=isch"

    def __init__(self):
        pass

    def get_images_url(self, keyword: str, nb_images: int) -> List[GrabbedImage]:
        query = keyword.split()
        query = '+'.join(query)
        url = self.GOOGLE_URL % query

        print('> searching image on Google : ' + url)

        options = webdriver.ChromeOptions()

        browser = webdriver.Chrome(options=options)

        browser.get(url)
        time.sleep(2)

        elem = browser.find_element(By.TAG_NAME, "body")

        # scroll to fire the infinite scroll event and load more images
        no_of_pages_down = 20 if nb_images < 300 else 100
        while no_of_pages_down:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
            no_of_pages_down -= 1
            try:
                show_more_btn = browser.find_element(By.ID, "smb")
                if show_more_btn.is_displayed():
                    show_more_btn.click()
            except Exception as e:
                pass

        # allow any lazy-loaded images to render
        time.sleep(1)


        images_objects = []
        if self.full_image:
            # Click each thumbnail and read the displayed large image(s) from the viewer
            # broaden selectors to handle variations of Google DOM
            thumbnails = browser.find_elements(By.CSS_SELECTOR, "img.rg_i, img.Q4LuWd")
            print(f"> thumbnails found: {len(thumbnails)}")
            found = 0
            for idx, thumb in enumerate(thumbnails):
                if found >= nb_images:
                    break
                try:
                    browser.execute_script("arguments[0].scrollIntoView(true);", thumb)
                    try:
                        thumb.click()
                    except Exception:
                        # fallback to JS click
                        browser.execute_script("arguments[0].click();", thumb)
                    time.sleep(0.8)
                    # large displayed images in the side panel have class 'n3VNCb' (or similar)
                    large_imgs = browser.find_elements(By.CSS_SELECTOR, "img.n3VNCb, img.irc_mi")
                    print(f"> thumb {idx}: large candidates: {len(large_imgs)}")
                    for li in large_imgs:
                        src = li.get_attribute('src')
                        if src and StringUtil.is_http_url(src):
                            image_obj = GrabbedImage()
                            image_obj.source = GrabSourceType.GOOGLE.value
                            image_obj.url = src
                            # guess extension from url
                            try:
                                ext = os.path.splitext(src.split('?')[0])[1].lstrip('.')
                                image_obj.extension = ext
                            except Exception:
                                image_obj.extension = ''
                            images_objects.append(image_obj)
                            found += 1
                            break
                except Exception as ex:
                    print(f"> error handling thumb {idx}: {ex}")
                    pass
        else:
            images = browser.find_elements(By.CSS_SELECTOR, "img.rg_i, img.Q4LuWd")
            for image in images:
                image_obj = GrabbedImage()
                image_obj.source = GrabSourceType.GOOGLE.value
                src = image.get_attribute('src')
                if StringUtil.is_http_url(src):
                    image_obj.url = src
                else:
                    image_obj.base64 = src
                images_objects.append(image_obj)

        browser.close()

        return images_objects
