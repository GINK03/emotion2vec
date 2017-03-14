
import sys
import json
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from bs4 import BeautifulSoup
import time
def scraping():
  cap = DesiredCapabilities.PHANTOMJS
  cap["phantomjs.page.settings.userAgent"] = "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36"
  cap["phantomjs.page.settings.javascriptEnabled"] = True
  cap["phantomjs.page.settings.loadImages"] = True
  url = "http://www.yahoo.com"
  driver = webdriver.PhantomJS('/usr/local/bin/phantomjs', desired_capabilities=cap)
  driver.set_window_size(1366, 2000)
  driver.get("http://www.google.co.jp")
  time.sleep(5.)
  html = driver.page_source
  print(html)
  soup = BeautifulSoup(html, "lxml")
  print(html)
  header = soup.find("head")
  title = header.find("title")
  #description = header.find("meta", attrs={"name": "description"})
  #description_content = description.attrs['content']
  #output = {"title": title, "description": description_content}
  t = soup.find('div', {'class': 'item-box-title'}).text
  print(t)
  #print(output)
if __name__ == '__main__':

  scraping()
