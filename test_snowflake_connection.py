from src.common.snowflake_client import SnowflakeClient

sf = SnowflakeClient()

result = sf.fetchall("SHOW TABLES")

for row in result:
    print(row)

sf.close()