FROM python:3.13-trixie

RUN apt update && apt upgrade -y && \
    apt install libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 \
    libcairo2 libasound2 build-essential python3-venv python3-pip -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p converter-specs-cal-app

WORKDIR /converter-specs-cal-app

COPY . /converter-specs-cal-app

RUN pip install --upgrade --no-cache-dir -r requirements.txt && \
    plotly_get_chrome -y

EXPOSE 7860

CMD ["/bin/bash", "-c", "streamlit run main.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS true --server.enableXsrfProtection true"]