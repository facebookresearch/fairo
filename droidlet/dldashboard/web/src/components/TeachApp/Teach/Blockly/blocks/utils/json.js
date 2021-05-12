/*
Copyright (c) Facebook, Inc. and its affiliates.
*/

// finds the closing curly brace to match the opening curly brace at the given index
// or -1 if none
const findIndexOfMatchingBracket = (str, firstBracketIndex) => {
  let seen = 0;
  if (!firstBracketIndex || str.charAt(firstBracketIndex) !== "{") return -1;
  for (let i = firstBracketIndex; i < str.length; i++) {
    const char = str.charAt(i);
    if (char === "{") {
      seen += 1;
    } else if (char === "}") {
      seen -= 1;
      if (seen === 0) return i;
    }
  }
  return -1; // didn't find a matching bracket
};

// adds commas between json objects that are concatenated together
// does not add comma after final json object
export const addCommasBetweenJsonObjects = (jsonString) => {
  let i = jsonString.indexOf("{");
  while (i >= 0 && i < jsonString.length) {
    const nextBracketIndex = findIndexOfMatchingBracket(jsonString, i);
    i = jsonString.indexOf("{", nextBracketIndex);
    if (i < 0) break;
    jsonString =
      jsonString.slice(0, nextBracketIndex + 1) +
      "," +
      jsonString.slice(nextBracketIndex + 1);
    i += 1;
  }
  return jsonString;
};

// given a string of concatenated json objects, returns the amount of
// concatenated json objects in the string
export const getNumberConcatenatedJsonObjects = (jsonString) => {
  let i = jsonString.indexOf("{");
  let numSeen = 0;
  while (i >= 0 && i < jsonString.length) {
    numSeen += 1;
    const nextBracketIndex = findIndexOfMatchingBracket(jsonString, i);
    if (nextBracketIndex < 0) break;
    i = jsonString.indexOf("{", nextBracketIndex);
  }
  return numSeen;
};
