//filter will reemit the data if cb(err,pass) pass is truthy

// reduce is more tricky
// maybe we want to group the reductions or emit progress updates occasionally
// the most basic reduce just emits one 'data' event after it has recieved 'end'


var through = require('through')


module.exports = split

//TODO pass in a function to map across the lines.

function split (matcher, mapper) {
  var soFar = ''
  if('function' === typeof matcher)
    mapper = matcher, matcher = null
  if (!matcher)
    matcher = '\n'

  return through(function (buffer) { 
    var stream = this
      , pieces = (soFar + buffer).split(matcher)
    soFar = pieces.pop()

    for (var i = 0; i < pieces.length; i++) {
      var piece = pieces[i]
      if(mapper) {
        piece = mapper(piece)
        if('undefined' !== typeof piece)
          stream.emit('data', piece)
      }
      else
        stream.emit('data', piece)
    }

    return true
  },
  function () {
    if(soFar)
      this.emit('data', soFar)  
    this.emit('end')
  })
}

