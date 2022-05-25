var EventEmitter = require('events').EventEmitter;
var inherits = require('util').inherits;

module.exports = OrderedEmitter;

function OrderedEmitter (opts) {
    EventEmitter.call(this);
    if (!opts) opts = {};
    
    this._eventQueue = {};
    this._next = {};
    this.options = opts;
}
inherits(OrderedEmitter, EventEmitter);

OrderedEmitter.prototype.reset = function (evName) {
    if (evName === undefined) {
        this._next = {};
    }
    else {
        this._next[evName] = 0;
    }
};

OrderedEmitter.prototype.emit = function (evName, obj) {
    var emit = function (args) {
        EventEmitter.prototype.emit.apply(this, args);
    }.bind(this, arguments);
    
    var queue = this._eventQueue;
    var next = this._next;
    var name = this.options.span ? '*' : evName;
    
    if (typeof obj === 'object' && obj !== null
    && typeof obj.order === 'number') {
        if (!next[name]) next[name] = 0;
        
        if (obj.order === next[name]) {
            next[name] ++;
            emit();
            
            while (queue[name] && queue[name][next[name]]) {
                queue[name][next[name]]();
                delete queue[name][next[name]];
                next[name] ++;
            }
        }
        else {
            if (!queue[name]) queue[name] = {};
            queue[name][obj.order] = emit;
        }
    }
    else emit()
};
