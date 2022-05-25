var spawn = require('child_process').spawn;
var through = require('through');
var split = require('event-stream').split;
var run = require('comandante');

exports = module.exports = function (ref, opts) {
    if (ref && typeof ref === 'object') {
        opts = ref;
        return {
            list : function (ref, file) { return readDir(ref, file, opts) },
            read : function (ref, file) { return readFile(ref, file, opts) },
        }
    }
    else if (opts) {
        return {
            list : function (file) { return readDir(ref, file, opts) },
            read : function (file) { return readFile(ref, file, opts) },
        }
    }
    else return {
        list : readDir.bind(null, ref),
        read : readFile.bind(null, ref),
    };
};

exports.list = readDir;
exports.read = readFile;

function show (ref, file, opts) {
    if (file === '.') file = './';
    return run('git', [ 'show', ref + ':' + file ], opts);
}

function readFile (ref, file, opts) {
    return show(ref, file, opts);
}

function readDir (ref, dir, opts) {
    var num = 0;
    var tr = through(function (line) {
        if (num === 0) {
            if (line !== 'tree ' + ref + ':' + dir
            && line !== 'tree ' + ref + ':' + dir + '/') {
                this.emit('error', ref + ':' + dir + ' is not a directory');
            }
        }
        else if (num === 1) {
            if (line !== '') {
                this.emit('error',
                    'unexpected data reading directory: ' + line
                );
            }
        }
        else this.emit('data', line);
        
        num ++;
    });
    return show(ref, dir, opts).pipe(split()).pipe(tr);
}
