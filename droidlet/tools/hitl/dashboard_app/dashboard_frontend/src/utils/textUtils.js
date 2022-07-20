/*
Copyright (c) Facebook, Inc. and its affiliates.

Utils for processing text
*/
export const toFirstCapital = (word) => (
    `${word.substring(0, 1).toUpperCase()}${word.substring(1).toLowerCase()}`
);

export const snakecaseToWhitespaceSep = (word) => {
    let splittedWords = word.split("_");
    splittedWords = splittedWords.map((w) => toFirstCapital(w));
    return splittedWords.join(" ")
};