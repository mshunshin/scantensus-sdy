from .sdy_file import SDYFile

class SDYProcessor:
    def prepare_for_inference(file: SDYFile):
        """
        Prepares the SDY file for inference.
        """
        pass


    def _get_shards(self, shard_width, shard_overlap, total_width):
        source_list = []
        destination_list = []

        source_start = 0
        source_end = shard_width

        destination_start = 0
        destination_end = shard_width - shard_overlap

        while destination_end <= total_width:
            source = (source_start, source_end)
            destination = (destination_start, destination_end)

            source_list.append(source)
            destination_list.append(destination)

            source_start = source_start + (shard_width - 2*shard_overlap)
            source_end = source_start + shard_width

            destination_end = source_end - shard_overlap
            destination_start = destination_end

            if source_end > total_width:
                source_end = total_width
                source_start = total_width - shard_width
                destination_end = total_width
                destination_start = total_width - shard_width + shard_overlap

                source = (source_start, source_end)
                destination = (destination_start, destination_end)

                source_list.append(source)
                destination_list.append(destination)
                break

        return source_list, destination_list
