FROM python:3.11
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
RUN mkdir -p $HOME/app/data/vectorstore && chown -R user:user $HOME/app/data
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]