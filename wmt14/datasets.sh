#!/bin/bash

# 设置目标目录
TARGET_DIR=~/Desktop/wmt14

# 创建目录（如果不存在）
mkdir -p $TARGET_DIR

# 切换到目标目录
cd $TARGET_DIR

# 下载文件并解压
echo "Downloading Europarl v7 English-German..."
wget http://www.statmt.org/europarl/v7/de-en.tgz
echo "Extracting de-en.tgz..."
tar -xzf de-en.tgz

echo "Downloading Common Crawl English-German..."
wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
echo "Extracting training-parallel-commoncrawl.tgz..."
tar -xzf training-parallel-commoncrawl.tgz

echo "Downloading News Commentary v9 English-German..."
wget http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
echo "Extracting training-parallel-nc-v9.tgz..."
tar -xzf training-parallel-nc-v9.tgz

echo "Downloading WMT14 test set..."
wget http://www.statmt.org/wmt14/test-full.tgz
echo "Extracting test-full.tgz..."
tar -xzf test-full.tgz

# 输出完成信息
echo "All files downloaded and extracted to $TARGET_DIR"
