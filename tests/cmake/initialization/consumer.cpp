#ifndef UNI20_FILL_UNINITIALIZED_SNAN
#error Initialization diagnostics did not propagate through the interface target
#endif
static_assert(UNI20_FILL_UNINITIALIZED_SNAN == EXPECTED_SNAN);
int main() { return 0; }
