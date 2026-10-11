#pragma once
#include "pch.h"

namespace Magpie {
class Logger {
public:
	static Logger& Get() { static Logger logger; return logger; }
	template<class... Args>
	void Win32Error(const char* message, Args&&...) {
		Fake::state.errors.emplace_back(message);
	}
	void Info(const std::string&) {}
};
}
