# Dockerfile for Frontend
FROM node:18-alpine

WORKDIR /app

# Copy dependency files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Command to start the application
CMD ["npm", "run", "dev"]