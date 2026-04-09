pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Repository checked out successfully.'
            }
        }

        stage('Sanity Check') {
            steps {
                sh 'pwd'
                sh 'ls -la'
            }
        }
    }
}
