# Dockerfile
FROM software-base:latest
# 暴露端口
EXPOSE 5000

# 启动命令

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--concurrency=4"]