# Changelog

## 2.4.0
- HashMap instances are now compatible ES6 iterators. Thanks @NathanJang

## 2.3.0
- Constructor now supports a 2D array 2D of key-value pair. Thanks @ulikoehler
- Renamed `remove()` to `delete()`. `remove()` is now deprecated and kept temporarily. Thanks @ulikoehler
- Added `.size` member. `count()` is now deprecated and kept temporarily. Thanks @ulikoehler

## 2.2.0
- Added entries() method to hashmaps. Thanks @ulikoehler

## 2.1.0
- support ECMA 5 non-conformant behaviour of Microsoft edge #27. Thanks @freddiecoleman

## 2.0.6
- Names of chained methods is hardcoded rather than using the "return" trick. Fixes bug when minified, thanks @fresheneesz.
- Added jshint to be run before any commit

## 2.0.5
- count() is now O(1), thanks @qbolec

## 2.0.4
- hasOwnProperty() is used to check for the internal expando, thanks @psionski

## 2.0.3
- forEach method accepts a context as 2nd argument, thanks @mvayngrib

## 2.0.2
- Make collisions rarer

## 2.0.1
- AMD CommonJS export is now compatible

## 2.0.0
- Added chaining to all methods with no returned value
- Added multi() method
- Added clone() method
- Added copy() method
- constructor accepts one argument for cloning or several for multi()

## 1.2.0
- Added search() method, thanks @rafalwrzeszcz

## 1.1.0
- AMD support, thanks @khrome

## 1.0.1
- forEach() callback receives the hashmap as `this`
- Added keys()
- Added values()

## 1.0.0
- First release
