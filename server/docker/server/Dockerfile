# Dockerfile
FROM software-base:latest


# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "-c", "gunicorn.conf.py", "run:app"]

