
import sys
import json
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import selenium
from bs4 import BeautifulSoup
import time
import http
import concurrent.futures
import urllib
SLEEP = 0.1
def tag_scrape():
  cap = DesiredCapabilities.PHANTOMJS
  cap["phantomjs.page.settings.userAgent"] = "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36"
  cap["phantomjs.page.settings.javascriptEnabled"] = True
  cap["phantomjs.page.settings.loadImages"] = True
  driver = webdriver.PhantomJS('/usr/local/bin/phantomjs', desired_capabilities=cap)
  cookies = json.loads(open('cookies.json', 'r').read())
  for cookie in cookies:
    driver.add_cookie(cookie)
  driver.set_window_size(1366, 2000)
  driver.get("http://qiita.com/tags")
  with open('tags.txt', 'w') as f:
    for _ in range(255):
      driver.find_element_by_class_name("js-next-page-link").click();
      time.sleep(1.)
      html  = driver.page_source
      soup  = BeautifulSoup(html, "html.parser")
      for a in soup.find_all('a', {'class': 'u-link-unstyled TagList__label'}):
        f.write( "http://qiita.com%s\n"%a['href'] ) 
        print( _, "http://qiita.com%s"%a['href'] ) 
    driver.save_screenshot("tag_scraping.png")

def _get_tags_links(taglink):
  cap = DesiredCapabilities.PHANTOMJS
  cap["phantomjs.page.settings.userAgent"] = "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36"
  cap["phantomjs.page.settings.javascriptEnabled"] = True
  cap["phantomjs.page.settings.loadImages"] = True
  driver = webdriver.PhantomJS('/usr/local/bin/phantomjs', desired_capabilities=cap)
  cookies = json.loads(open('cookies.json', 'r').read())
  for cookie in cookies:
    driver.add_cookie(cookie)
  driver.set_window_size(1366, 2000)
  with open('each_tags_link_%s.txt'%(taglink.split('/')[-1]), 'w') as f:
    driver.get(taglink + '/items')
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    one_to_x = soup.find('li', {'class': 'disabled hidden-lg hidden-sm hidden-md'}).text
    last = int(one_to_x.strip().split(' of ')[-1])
    for a in soup.find_all('a', {'class': 'u-link-no-underline'}):
      f.write( "http://qiita.com%s\n"%a['href'] ) 
      print( "http://qiita.com%s"%a['href'] ) 
    #sys.exit()
    for _ in range(1,last):
      try:
        driver.find_element_by_class_name("js-next-page-link").click();
      except selenium.common.exceptions.NoSuchElementException as e:
        break
      except http.client.RemoteDisconnected as e:
        continue
      except urllib.error.URLError as e:
        continue
      time.sleep(SLEEP)
      html = driver.page_source
      soup = BeautifulSoup(html, "html.parser")
      for a in soup.find_all('a', {'class': 'u-link-no-underline'}):
        f.write( "http://qiita.com%s\n"%a['href'] ) 
        print( _, taglink, "http://qiita.com%s"%a['href'] ) 
    driver.save_screenshot("tag_scraping.png")

def each_tags_link_scrape():
  with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    taglinks = [taglink for taglink in filter(lambda x:x!='', open('tags.txt', 'r').read().split('\n'))]
    for taglink in taglinks:
      executor.submit(_get_tags_links, taglink)

def _get_content(urls):
  cap = DesiredCapabilities.PHANTOMJS
  cap["phantomjs.page.settings.userAgent"] = "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36"
  cap["phantomjs.page.settings.javascriptEnabled"] = True
  cap["phantomjs.page.settings.loadImages"] = True
  driver = webdriver.PhantomJS('/usr/local/bin/phantomjs', desired_capabilities=cap)
  cookies = json.loads(open('cookies.json', 'r').read())
  for cookie in cookies:
    driver.add_cookie(cookie)
  driver.set_window_size(1366, 2000)
  for e,url in enumerate(urls):
    entities = url.split('/')
    file_name = 'contents/%s_%s.txt'%(entities[-3], entities[-1])
    from pathlib import Path 
    if Path(file_name).is_file():
      continue

    with open(file_name, 'w') as f:
      driver.get(url)
      html = driver.page_source
      soup = BeautifulSoup(html, "html.parser")
      context =  soup.find('section', {'class': 'markdownContent'} ).text 
      tags    =  list(map(lambda x:x.text, soup.find_all('li', {'class': 'TagList__item'})))
      f.write(json.dumps({'tags':tags, 'context':context}))
    print('finished e=%d %s'%(e, url))

def get_content():
  import glob
  allurls = list(set(sum([open(name).read().split('\n') for name in glob.glob('each_tags_link/*')], [])))
  print(len(allurls))
  # make random bucket
  buckets = []
  for i in range(0, len(allurls) - 100, 100):
    buckets.append(allurls[i:i+100])
  with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    for id, bucket in enumerate(buckets):
      executor.submit(_get_content, bucket)

def init_cookies():
  cap = DesiredCapabilities.PHANTOMJS
  cap["phantomjs.page.settings.userAgent"] = "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.66 Safari/537.36"
  cap["phantomjs.page.settings.javascriptEnabled"] = True
  cap["phantomjs.page.settings.loadImages"] = True
  #cookies = json.loads(open('cookies.json', 'r').read())
  driver = webdriver.PhantomJS('/usr/local/bin/phantomjs', desired_capabilities=cap)
  #for cookie in cookies:
  #  driver.add_cookie(cookie)
  driver.set_window_size(1366, 2000)
  driver.get("http://qiita.com/items")
  identity = driver.find_element_by_id("identity")
  identity.send_keys("nardtree")
  password = driver.find_element_by_id("password")
  password.send_keys("******")
  driver.find_element_by_name("commit").click()
  driver.save_screenshot("test.png")
  html  = driver.page_source
  soup  = BeautifulSoup(html, "html.parser")
  title = soup.find("title")
  open('cookies.json', 'w').write( json.dumps(driver.get_cookies(), indent=4) )
  print( json.dumps(driver.get_cookies(), indent=4) )

  #print(output)
if __name__ == '__main__':
  if '--init' in sys.argv:
    init_cookies()
  if '--scrape' in sys.argv:
    tag_scrape()
  if '--each_scrape' in sys.argv:
    each_tags_link_scrape()
  if '--contents' in sys.argv:
    get_content()
