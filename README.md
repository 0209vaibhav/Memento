# Design in Action

## Setup Instructions

### Firebase Configuration

1. Create a new Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable the following Firebase services:
   - Authentication
   - Firestore Database
   - Storage
   - Analytics (optional)
3. Get your Firebase configuration:
   - Go to Project Settings
   - Scroll down to "Your apps"
   - Click on the web app icon (</>)
   - Register your app if you haven't already
   - Copy the configuration object
4. Set up your Firebase configuration:
   - Copy `scripts/firebase-config.template.js` to `scripts/firebase-config.js`
   - Replace the placeholder values in `firebase-config.js` with your actual Firebase configuration
   - Never commit `firebase-config.js` to version control

### Environment Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## Security Notes

- Never commit sensitive information like API keys or credentials to version control
- Keep your Firebase configuration file (`firebase-config.js`) local and add it to `.gitignore`
- Use environment variables for sensitive configuration in production
- Regularly rotate API keys and credentials

## Development

- The project uses modern web technologies and follows best practices for security
- All sensitive files are listed in `.gitignore`
- Use the template files provided for configuration setup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Remember to never commit sensitive information or API keys in your pull requests. 