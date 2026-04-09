pipeline {
    agent any

    environment {
        IMAGE_NAME = "luna16-roi-app:latest"
        CONTAINER_NAME = "luna16-roi-container"
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Repository checked out successfully.'
                sh 'pwd'
                sh 'ls -la'
            }
        }

        stage('Sanity Checks') {
            steps {
                sh '''
                python3 -m venv .venv
                . .venv/bin/activate
                pip install --upgrade pip
                if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
                pip install black ruff pytest
                black --check . || true
                ruff check . || true
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                . .venv/bin/activate
                if [ -d tests ]; then pytest || true; else echo "No tests directory yet"; fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('Container Smoke Test') {
            steps {
                sh '''
                docker rm -f smoke-test || true
                docker run -d --name smoke-test -p 8001:8000 $IMAGE_NAME
                sleep 10
                curl -f http://127.0.0.1:8001/health || (docker logs smoke-test && exit 1)
                docker rm -f smoke-test
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                docker rm -f $CONTAINER_NAME || true
                docker run -d --name $CONTAINER_NAME -p 8000:8000 --restart unless-stopped $IMAGE_NAME
                '''
            }
        }
    }
}
