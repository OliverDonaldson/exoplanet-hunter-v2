# Frontend image: Vite production build served by nginx.
# Build from the repository root:
#   docker build -f docker/frontend.Dockerfile -t exoplanet-hunter-frontend .
FROM node:20-slim AS build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /build/dist /usr/share/nginx/html
# /api/* is proxied to the api service; compose provides the network alias.
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    location /api/ {\n\
        proxy_pass http://api:8000/;\n\
    }\n\
    location / {\n\
        try_files $uri /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 80
