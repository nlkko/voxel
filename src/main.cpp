#include "engine/engine.hpp"

int main(int argc, char* argv[])
{
	// Controlling lifetime of engine
	{
		Engine engine;
		engine.run();
	}

	return 0;
}