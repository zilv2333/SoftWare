#这个模板可以直接使用。

from datetime import timedelta

from .api.mysql import DatabaseConfig
from .api import opengauss
import os
JWT_SECRET_KEY='your-secret-key-change-in-production'
JWT_ACCESS_TOKEN_EXPIRES=timedelta(hours=24)
SECRET_KEY='your-secret-key-change-in-production'
JWT_ALGORITHM='HS256'

MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB 限制
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

DIFY_API_URL=os.getenv('DIFY_API_URL','')
DIFY_API_KEY=os.getenv('DIFY_API_KEY','')
VIDEO_FOLDER='example/video'
THUMBNAIL_FOLDER='example/thumbnail'
env = os.getenv('FLASK_ENV', 'development')
DataBase_Name = os.getenv('DATABASE', 'mysql')
REDIS_HOST=os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT=os.getenv('REDIS_PORT', 6379)



if env == 'production':

    MySql_db_config = DatabaseConfig(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'xyc'),
        password=os.getenv('MYSQL_PASSWORD', '123456'),
        database=os.getenv('MYSQL_DATABASE', 'software'),
        port=int(os.getenv('MYSQL_PORT', 3306)),
    )


else:
    MySql_db_config = DatabaseConfig(
        host='localhost',
        user='xyc',
        password='123456',  # 修改为您的MySQL密码
        database='software'
    )


OG_db_config = opengauss.DatabaseConfig(
        host=os.getenv('GS_HOST','localhost'),
        user=os.getenv('GS_USERNAME'),
        password=os.getenv('GS_PASSWORD'),
        database=os.getenv('GS_DATABASE'),
        port=os.getenv('GS_PORT')
    )


