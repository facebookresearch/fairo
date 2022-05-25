module.exports = function (d) {
    var tzo = 0, m;
    if (m = /([-+]\d+)($|\s*\(\S+\)$)/.exec(d)) {
        tzo = Number(m[1]);
        tzo = -(Math.floor(tzo / 100) * 60 + (tzo % 100));
    }
    if (typeof d === 'string') d = new Date(d);
    
    if (tzo) {
        d = new Date(d.valueOf() - 1000 * (tzo - d.getTimezoneOffset()) * 60);
    }
    else tzo = d.getTimezoneOffset()
    
    var month = pad(d.getMonth() + 1, 2);
    var date = pad(d.getDate(), 2);
    var ymd = [ d.getFullYear(), month, date ].join('-');
    
    var h = pad(d.getHours(), 2);
    var m = pad(d.getMinutes(), 2);
    var s = pad(d.getSeconds(), 2);
    var hms = [ h, m, s ].join(':');
    
    var tzs = tzo > 0 ? '-' : '+';
    var tzh = tzs + pad(Math.floor(Math.abs(tzo) / 60), 2);
    var tzm = pad(Math.abs(tzo) % 60, 2);
    return ymd + 'T' + [h,m,s].join(':') + [tzh,tzm].join(':');
};

function pad (x, n) {
    return (Array(n).join('0') + String(x)).split('').slice(-n).join('');
}
