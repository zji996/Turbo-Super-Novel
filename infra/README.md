# infra

基础设施目录（Docker / K8s / Terraform）。

## Dev compose

在 repo_root 执行：

```bash
docker compose -f infra/docker-compose.dev.yml up -d
```

包含：

- Redis（Celery broker/result backend）
- MinIO（S3 兼容对象存储，默认 Console: http://localhost:9001）
- Postgres（默认 `postgres/postgres`，DB: `tsn`，端口 `5432`）
