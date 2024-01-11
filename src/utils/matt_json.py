import logging

from typing import List
import datetime

def curve_list_to_str(curve_list: List, round_digits=1):
    out = " ".join([str(round(value, round_digits)) for value in curve_list])
    return out

def get_labels_from_firebase_project_data_new(project_data, project, mapping_data, EXPORT_FORMAT='backup'):
    logging.info(f'Project: {project}')
    # project_data = firebase_data['fiducial'][project]['labels']

    ACCEPT_TYPE_PROJECT = False

    db = []
    logging.info(f"Project {project} has {len(project_data)} labels")
    for image_name, image_data in project_data.items():

        if image_name[0].isnumeric() and image_name[1].isnumeric() and image_name[2] == "-":
            file = image_name.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")
            file = file.replace(":", ".")
        else:
            file_name_mangled_split = image_name.split('~')
            project_id = file_name_mangled_split[0]
            naming_scheme = file_name_mangled_split[1]

            if naming_scheme == 'unique':
                file = file_name_mangled_split[2].replace(":", ".")
                file = file.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")
            elif naming_scheme == 'clusters':
                file = image_name.replace(":", ".")
            else:
                logging.warning("Unrecognised file naming scheme")

        user_dict = {}

        try:
            for user_item in image_data['events'].items():
                try:
                    user_code = user_item[0]
                    try:
                        user = user_item[1]['user']
                    except Exception as e:
                        user = "unknown"
                    time_stamp = user_item[1]['t']

                    user_data = {}
                    user_data['user'] = user
                    user_data['time_stamp'] = time_stamp

                    user_dict[user_code] = user_data
                except Exception:
                    logging.exception("Error in User events")
                    print(f"user_item: {user_item}")
        except Exception:
            ACCEPT_TYPE_PROJECT = True

        nodes = image_data.get('nodes', {})
        curves = image_data.get('curves', {})

        labels = {**nodes, **curves}

        for label_name_old, label_data_all in labels.items():

            label_name = mapping_data.get(label_name_old, None)

            if label_name is None:
                logging.warning(f"Missing mapping for {label_name_old}, using old name")
            
                label_name = label_name_old

            for user_code, node_data in label_data_all.items():
                if not ACCEPT_TYPE_PROJECT:
                    true_user = user_dict[user_code]['user']
                    true_time_stamp = user_dict[user_code]['time_stamp']
                else:
                    true_user = user_code
                    true_time_stamp = datetime.datetime.utcnow().isoformat() + 'Z'

                if 'format' in node_data:
                    DATABASE_FORMAT = node_data['format']
                else:
                    DATABASE_FORMAT = 1
                    logging.warning(f"Database format is old: {DATABASE_FORMAT}, {label_name}, converting")
                    fake_node_data = {}
                    if type(node_data) is list:
                        if len(node_data) > 0:
                            fake_node_data['vis'] = 'seen'
                        else:
                            fake_node_data['vis'] = 'blurred'
                    elif type(node_data) is dict:
                        if node_data.get('0'):
                            fake_node_data['vis'] = 'seen'
                        else:
                            fake_node_data['vis'] = 'blurred'

                    fake_node_data['instances'] = {}
                    fake_node_data['instances']['0'] = {}

                    if type(node_data) is dict:
                        if node_data.get('0'):
                            logging.info("nodes")
                            fake_node_data['instances']['0']['nodes'] = node_data.copy()
                        elif node_data.get('x'):
                            logging.info("single node")
                            fake_node_data['instances']['0']['node'] = node_data.copy()
                        else:
                            raise Exception(f"error in converting from type 1 to type 2 database format")
                    elif type(node_data) is list:
                        if len(node_data) > 1:
                            logging.info("nodes")
                            fake_node_data['instances']['0']['nodes'] = node_data.copy()
                        elif len(node_data) == 1:
                            logging.info("single node")
                            fake_node_data['instances']['0']['node'] = node_data.copy()
                        else:
                            raise Exception(f"error in converting from type 1 to type 2 database format")

                    node_data = fake_node_data
                    DATABASE_FORMAT = 2

                vis = node_data.get('vis', 'seen')

                if vis == "unasked":
                    continue

                if vis == "blurred":
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'instance_num': 0,
                           'vis': 'blurred',
                           'value_x': '',
                           'value_y': '',
                           'straight_segment': '',
                           }

                    db.append(out)
                    continue

                if vis == 'off':
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'instance_num': 0,
                           'vis': 'off',
                           'value_x': '',
                           'value_y': '',
                           'straight_segment': '',
                           }
                    db.append(out)
                    continue

                try:
                    if DATABASE_FORMAT == 2:
                        freehand_node = False
                        node_data_instances = node_data.get('instances')

                        if type(node_data_instances) is list:
                            node_data_instances_dict = {str(a): b for a, b in enumerate(node_data_instances)}
                        elif type(node_data_instances) is dict:
                            node_data_instances_dict = node_data_instances

                        for node_instance_num, node_instance in node_data_instances_dict.items():

                            if node_instance.get('isFreehand'):
                                freehand_node = True
                                single_node = False
                                try:
                                    node_list = node_instance['freehandPoints']
                                except:
                                    logging.exception(f"No freehandPoints despite freehand")
                                    continue

                            elif 'node' in node_instance:
                                single_node = True
                                node_list = node_instance['node']

                            elif 'nodes' in node_instance:
                                single_node = False
                                node_list = node_instance['nodes']

                            else:
                                single_node = True
                                node_list = node_instance

                            curve_x = []
                            curve_y = []
                            curve_next_straight = []

                            if not single_node:

                                if type(node_list) is list:
                                    node_list_dict = {str(a): b for a, b in enumerate(node_list)}
                                elif type(node_list) is dict:
                                    node_list_dict = node_list

                                keys = sorted([int(x) for x in node_list_dict.keys()])
                                for key in keys:
                                    node = node_list_dict[str(key)]
                                    curve_x.append(node['x'])
                                    curve_y.append(node['y'])
                                    if not freehand_node:
                                        curve_next_straight.append(node.get("straightToNext", False))
                                    else:
                                        curve_next_straight.append(node.get("straightToNext", False))

                            elif single_node:
                                try:
                                    curve_x.append(node_list['x'])
                                    curve_y.append(node_list['y'])
                                    curve_next_straight.append(False)
                                except Exception as e:
                                    logging.exception("Error in single node")

                            curve_next_straight = [1 if x else 0 for x in curve_next_straight]

                            if freehand_node:
                                target_num_points = 50
                                num_points = len(curve_x)

                                if num_points > target_num_points:
                                    step = (num_points - 1.0) / (target_num_points - 1.0)
                                    select_idx = [round(i*step) for i in list(range(target_num_points))]

                                    curve_x = [curve_x[i] for i in select_idx]
                                    curve_y = [curve_y[i] for i in select_idx]
                                    curve_next_straight = [curve_next_straight[i] for i in select_idx]

                            if len(curve_x) == 0 or len(curve_y) == 0:
                                out_vis = "blurred"
                                logging.error(f"Unexpected empty curve_x when seen {project} {file} {true_user} {label_name} {node_instance_num}")
                            else:
                                out_vis = "seen"

                            out = {'project': project,
                                   'file': file,
                                   'user': true_user,
                                   'time': true_time_stamp,
                                   'label': label_name,
                                   'vis': out_vis,
                                   'instance_num': int(node_instance_num),
                                   'value_x': curve_list_to_str(curve_x, round_digits=2),
                                   'value_y': curve_list_to_str(curve_y, round_digits=2),
                                   'straight_segment': curve_list_to_str(curve_next_straight),
                                   }

                            db.append(out)

                except Exception:
                    print("d")
                    logging.exception(f"Failure in project data")

    return db
