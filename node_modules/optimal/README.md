# Optimal

[![Build Status](https://travis-ci.org/milesj/optimal.svg?branch=master)](https://travis-ci.org/milesj/optimal)
[![npm version](https://badge.fury.io/js/optimal.svg)](https://www.npmjs.com/package/optimal)
[![npm deps](https://david-dm.org/milesj/optimal.svg)](https://www.npmjs.com/package/optimal)

A system for building and validating defined object structures, like argument options, configuration
files, data bags, validation fields, and more! Runs checks in development, and strips checks in
production using dead code elimination.

- Recursively builds and validates nested structures.
- Supports common data types.
- Autofills missing fields with default values.
- Allows or restricts unknown fields.
- Mark fields as nullable or required.
- Handles complex operators like AND, OR, and XOR.

## Requirements

- Node 8.9 (server)
- IE 11+ (browser)

## Installation

```
yarn add optimal
// OR
npm install optimal --save
```

## Documentation

- [Optimal](#optimal)
  - [Blueprint](#blueprint)
  - [Customization](#customization)
- [Predicates](#predicates)
  - [Array](#array)
  - [Bool](#bool)
  - [Custom](#custom)
  - [Date](#date)
  - [Function](#func)
  - [Instance](#instance)
  - [Number](#number)
  - [Object](#object)
  - [Regex](#regex)
  - [Shape](#shape)
  - [String](#string)
  - [Union](#union)

Will write eventually...
