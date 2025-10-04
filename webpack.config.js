const path = require('path');
module.exports = {
    mode: 'development',
    entry: './typescript/app.ts',
    output: {
        filename: 'app.bundle.js',
        path: path.resolve(__dirname, 'web/static/js'),
        publicPath: '/static/',
    },
    resolve: {
        extensions: ['.ts', '.js'],
        fallback: {
            'fs': false,
            'path': false,
            'crypto': false,
            'util': false,
            'stream': false,
            'buffer': false
        }
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
        ],
    },
    watch: true,
    watchOptions: {
        ignored: /node_modules/,
    },
    externals: {
        'onnxruntime-web': 'ort'
    },
};