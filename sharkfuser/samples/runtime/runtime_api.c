#include <iree/hal/api.h>
#include <iree/hal/drivers/init.h>

#include <stdio.h>

int main(int argc, char **argv) {
  iree_hal_driver_registry_t *registry = iree_hal_driver_registry_default();
  IREE_CHECK_OK(iree_hal_register_all_available_drivers(registry));

  iree_host_size_t driver_count;
  iree_hal_driver_info_t *driver_infos;
  IREE_CHECK_OK(iree_hal_driver_registry_enumerate(
      registry, iree_allocator_system(), &driver_count, &driver_infos));

  for (iree_host_size_t i = 0; i < driver_count; ++i) {
    printf("Available driver: %.*s (%.*s)\n",
           (int)driver_infos[i].driver_name.size,
           driver_infos[i].driver_name.data,
           (int)driver_infos[i].full_name.size, driver_infos[i].full_name.data);
  }
  return 0;
}
