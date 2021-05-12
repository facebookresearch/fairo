// -*- Mode: c++; c-basic-offset: 4; tab-width: 4; -*-

#include "tclap/CmdLine.h"
#include "tclap/DocBookOutput.h"
#include "tclap/ZshCompletionOutput.h"
#include <iostream>
#include <string>

using namespace TCLAP;
using namespace std;

int main(int argc, char** argv)
{
	CmdLine cmd("this is a message", ' ', "0.99" );
	DocBookOutput docoutput;
	ZshCompletionOutput zshoutput;
	CmdLineOutput *output = &zshoutput;

	if (argc > 2)
		output = &docoutput;

	cmd.setOutput(output);

	SwitchArg btest("B","sB", "exist Test B", false);
	MultiArg<int> atest("A","sA", "exist Test A", false, "integer");

	ValueArg<string> stest("s", "Bs", "string test", true, "homer",
						   "string");

	cmd.xorAdd(stest, btest);
	cmd.add( atest );

	cmd.parse(argc,argv);
}



